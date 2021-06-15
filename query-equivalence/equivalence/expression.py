import copy
import functools

import pglast
from pglast.enums import BoolExprType, A_Expr_Kind


def get_ast(expr):
    """Get the AST of the SQL expression by placing it within query and parsing it

    :param expr: expression string
    :return: expression AST
    """
    parsed = pglast.parse_sql(f"select {expr}")
    dict_query = (
        0,
        "RawStmt",
        "stmt",
        "SelectStmt",
        "targetList",
        0,
        "ResTarget",
        "val",
    )

    for key in dict_query:
        parsed = parsed[key]

    return parsed


def from_ast(ast_dict):
    """Builds an expression from a PostgreSQL AST

    :param ast_dict: AST dictionary
    :return: built expression
    """

    def traverse(node_dict):
        if len(node_dict.keys()) != 1:
            raise ValueError(
                "expression AST node must contain exactly one key"
            )

        node_type = (*node_dict,)[0]
        node = node_dict[node_type]

        if node_type == "BoolExpr":
            operator_name = node["boolop"]
            arguments = node["args"]

            if operator_name == BoolExprType.NOT_EXPR:
                if len(arguments) != 1:
                    raise ValueError(
                        "NOT operator must take exactly one argument"
                    )

                return Not(traverse(arguments[0]))

            operator_map = {
                BoolExprType.AND_EXPR: And,
                BoolExprType.OR_EXPR: Or,
            }

            operator_constructor = operator_map[operator_name]

            # convert plain n-ary operator to left-folded
            return functools.reduce(
                operator_constructor,
                map(traverse, arguments[1:]),
                traverse(arguments[0]),
            )

        if node_type == "ColumnRef":
            fields = node["fields"]

            if len(fields) == 1:
                column_name = fields[0]["String"]["str"]

                return ColumnRef(column_name)

            if len(fields) == 2:
                relation = fields[0]["String"]["str"]
                column_name = fields[1]["String"]["str"]

                return ColumnRef(column_name, relation)

            raise ValueError("column reference with >2 fields")

        if node_type == "A_Expr":
            kind = node["kind"]

            arguments = []

            if "lexpr" in node:
                arguments.append(node["lexpr"])

            if "rexpr" in node:
                arguments.append(node["rexpr"])

            if kind == A_Expr_Kind.AEXPR_OP:
                operator_name = node["name"][0]["String"]["str"]
                operator_map = {
                    # comparison
                    "=": Equal,
                    "!=": lambda a, b: Not(Equal(a, b)),
                    "<>": lambda a, b: Not(Equal(a, b)),
                    "<": Less,
                    "<=": LessOrEqual,
                    ">": Greater,
                    ">=": GreaterOrEqual,
                    # arithmetic
                    "+": Add,
                    "-": Subtract,
                    "*": Multiply,
                    "/": Divide,
                    "^": Power,
                }

                if operator_name not in operator_map:
                    raise ValueError(f"unknown operator: {operator_name}")

                return operator_map[operator_name](*map(traverse, arguments))

            raise ValueError(f"unknown A_Expr kind: {kind}")

        if node_type == "A_Const":
            constant_type = (*node["val"],)[0]

            if constant_type == "Null":
                return Null()

            constant_container = node["val"][constant_type]

            if constant_type == "Integer":
                return Const(constant_container["ival"])

            if constant_type == "Float":
                return Const(float(constant_container["str"]))

            if constant_type == "String":
                return Const(constant_container["str"])

            raise ValueError(f"unknown constant type: {constant_type}")

        if node_type == "NullTest":
            argument = traverse(node["arg"])
            null_test_type = node["nulltesttype"]

            if null_test_type == 0:
                return IsNull(argument)

            if null_test_type == 1:
                return Not(IsNull(argument))

            raise ValueError(f"unknown null test type: {null_test_type}")

        if node_type == "CaseExpr":
            cases = []
            for cases_node in node["args"]:
                condition = cases_node["CaseWhen"]["expr"]
                result = cases_node["CaseWhen"]["result"]

                cases.append((traverse(condition), traverse(result)))

            if "defresult" in node:
                default_expr = traverse(node["defresult"])
            else:
                default_expr = None

            if "arg" in node:
                test_expr = traverse(node["arg"])
            else:
                test_expr = None

            return Case(cases, default_expr, test_expr)

        if node_type == "FuncCall":
            function_name = node["funcname"][0]["String"]["str"]
            arguments = node.get("args", [])

            return Function(function_name, list(map(traverse, arguments)))

        if node_type == "TypeCast":
            child = traverse(node["arg"])

            if isinstance(child, Const):
                const_type = node["typeName"]["TypeName"]["names"][-1][
                    "String"
                ]["str"]
                cast_map = {
                    "int4": int,
                    "int8": int,
                    "text": str,
                    "bool": bool,
                }

                if const_type not in cast_map:
                    raise ValueError(f"unsupported cast to type {const_type}")

                return Const(cast_map[const_type](child.value))

            if isinstance(child, Null):
                return Null()

            raise ValueError(
                f"casting expressions of type {type(child).__name__} is unsupported"
            )

        if node_type == "BooleanTest":
            boolean_test_type = node["booltesttype"]
            boolean_test_map = {0: True, 1: False, 2: False, 3: True}
            child = traverse(node["arg"])

            if boolean_test_type not in boolean_test_map:
                raise ValueError(
                    f"unsupported boolean test type: {boolean_test_type}"
                )

            return Equal(Const(boolean_test_map[boolean_test_type]), child)

        raise ValueError(f"unknown expression node type: {node_type}")

    return simplify_nulls(traverse(ast_dict))


def from_str(s):
    if s.upper() == "FALSE":
        return Const(False)

    if s.upper() == "TRUE":
        return Const(True)

    if s.upper() == "NULL":
        return Null()

    return from_ast(get_ast(s))


def extract_column_refs(expr):
    """Extracts all column references from the expression

    :param expr: input expression
    :return: list of column references
    """
    column_refs = []

    def traverse(node):
        if isinstance(node, ColumnRef):
            column_refs.append(node)

        if isinstance(node, BinOp):
            traverse(node.left_expr)
            traverse(node.right_expr)

        if isinstance(node, Not):
            traverse(node.expr)

        if isinstance(node, Case):
            traverse(node.test_expr)
            traverse(node.default_expr)
            for case_cond, case_expr in node.cases:
                traverse(case_cond)
                traverse(case_expr)

        if isinstance(node, IsNull):
            traverse(node.child)

    traverse(expr)

    return column_refs


def extract_referenced_relations(expr):
    """Extracts all relations that are referenced in the expression

    :param expr: input expression
    :return: list of referenced relations names
    """
    expr_column_refs = extract_column_refs(expr)

    return [column_ref.relation for column_ref in expr_column_refs]


def simplify_nulls(expr):
    """Trivially simplifies expressions with NULL values

    :param expr: input expression
    :return: simplified expression
    """
    expr_copy = copy.deepcopy(expr)

    def traverse(node):
        if isinstance(node, BinOp):
            new_left_expr = traverse(node.left_expr)
            new_right_expr = traverse(node.right_expr)

            if isinstance(new_left_expr, Null) or isinstance(
                new_right_expr, Null
            ):
                return Null()

        if isinstance(node, Case):
            new_test_expr = traverse(node.test_expr)

            if isinstance(new_test_expr, Null):
                return Null()

        return node

    return traverse(expr_copy)


class Null:
    def __str__(self):
        return "null"


class Const:
    """Constant value of an arbitrary type"""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        if type(self.value) is bool:
            return str(self.value).lower()

        if type(self.value) is str:
            return f'"{self.value}"'

        return str(self.value)


class ColumnRef:
    """Reference to a column in a relation or subquery with alias"""

    def __init__(self, name, relation=None):
        self.name = name
        self.relation = relation

    def __str__(self):
        if self.relation is not None:
            return f"{self.relation}_{self.name}"
        else:
            return self.name


class BinOp:
    def __init__(self, left_expr, right_expr):
        self.left_expr = left_expr
        self.right_expr = right_expr

    @property
    def op(self):
        raise NotImplementedError

    def __str__(self):
        return f"({self.op} {self.left_expr} {self.right_expr})"


class Not:
    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return f"(not {self.expr})"


class And(BinOp):
    op = "and"


class Or(BinOp):
    op = "or"


class Equal(BinOp):
    op = "="


# class NotEqual(BinOp):
#     op = "!="


class Less(BinOp):
    op = "<"


class LessOrEqual(BinOp):
    op = "<="


class Greater(BinOp):
    op = ">"


class GreaterOrEqual(BinOp):
    op = ">="


class Add(BinOp):
    op = "+"


class Subtract(BinOp):
    op = "-"


class Multiply(BinOp):
    op = "*"


class Divide(BinOp):
    op = "/"


class Power(BinOp):
    op = "^"


class Function:
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __str__(self):
        return f"({self.name} {' '.join(map(str, self.args))})"


class IsNull:
    def __init__(self, child):
        self.child = child

    def __str__(self):
        return f"(not (= {self.child} {self.child}))"


class Case:
    def __init__(self, cases, default_expr=None, test_expr=None):
        self.cases = cases
        self.default_expr = default_expr
        self.test_expr = test_expr

    def __str__(self):
        result = self.default_expr

        for cond, expr in self.cases[::-1]:
            result = f"(ite {cond} {expr} {result})"

        return result
