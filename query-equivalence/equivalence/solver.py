import z3

from equivalence import expression, values


class Solver:
    def __init__(self, schema_info):
        self.smt_solver = z3.Solver()

        self.alias_map = {}
        self.schema_info = schema_info

        self.declarations = set()
        self.assertions = set()

    def compare_expressions(self, expr1, expr2):
        """Compares two expressions by checking their equivalence

        :param expr1: first expression
        :param expr2: second expression
        :return: True, iff expr1 is equivalent to the expr2
        """
        self.smt_solver.reset()

        return self.check_equivalence(expr1, expr2)

    def compare_expression_strings(self, s1, s2):
        """Compares two expression strings by checking expressions equivalence

        :param s1: first expression string
        :param s2: second expression string
        :return: True, iff s1 expression is equivalent to the s2 expression
        """
        return self.compare_expressions(
            expression.from_str(s1), expression.from_str(s2)
        )

    def add_assertion_str(self, assertion_str):
        """Updates the solver's context with a new arbitrary assertion

        :param assertion_str: assertion s-expression
        """
        self.assertions.add(assertion_str)

    def add_alias(self, relation, alias):
        """Updates the solver's alias map with a new relation alias

        :param relation: relation name
        :param alias: relation alias
        """
        self.alias_map[alias] = relation
        self.alias_map[relation] = relation

    def add_column_ref(self, column_ref: expression.ColumnRef):
        """Updates the solver's context with the definition of the column reference

        :param column_ref: column reference
        """
        type_map = {
            "integer": "Int",
            "int": "Int",
            "text": "String",
            "bool": "Bool",
        }

        if column_ref.relation is None:
            return

        if column_ref.relation not in self.alias_map:
            self.alias_map[column_ref.relation] = column_ref.relation

        relation = column_ref.relation
        while relation != self.alias_map[relation]:
            relation = self.alias_map[relation]

        column_type = type_map[
            self.schema_info[relation][column_ref.name].column_type
        ]

        self.declarations.add(f"(declare-const {column_ref} {column_type})")

        source_column_ref = expression.ColumnRef(column_ref.name, relation)
        if relation != column_ref.relation:
            self.declarations.add(
                f"(declare-const {source_column_ref} {column_type})"
            )
            self.assertions.add(
                f"(assert (= {column_ref} {source_column_ref}))"
            )

    def add_values_context(self, relation, rows):
        """Updates the solver's context with the definition of the value relation

        :param relation: values relation alias
        :param rows: relation rows
        """
        values_declarations, values_assertions = values.solver_context(
            relation, rows
        )

        self.declarations |= values_declarations
        self.assertions |= values_assertions

    def check_equivalence(self, expr1, expr2):
        """Checks that two expressions are equivalent

        :param expr1: first expression to check
        :param expr2: second expression to check
        :return: True, iff expr1 is equivalent to expr2
        """
        for column_ref in expression.extract_column_refs(expr1):
            self.add_column_ref(column_ref)

        for column_ref in expression.extract_column_refs(expr2):
            self.add_column_ref(column_ref)

        separator = "\n"

        program = separator.join(self.declarations)
        program += separator
        program += separator.join(
            self.assertions | {f"(assert (not (= {expr1} {expr2})))"}
        )

        self.smt_solver.from_string(program)

        if not self.smt_solver.check() == z3.unsat:
            return False

        return True

    def check_validity(self, expr):
        """Checks whether the given logical expression is true

        :param expr: an expression to check
        :return: True, iff expr is true
        """
        return self.check_equivalence(expr, expression.Const(True))

    def check_falsity(self, expr):
        """Checks whether the given logical expression is false

        :param expr: an expression to check
        :return: True, iff expr is false
        """
        return self.check_equivalence(expr, expression.Const(False))

    def is_constant(self, expr):
        """Checks whether the given expression is constant

        :param expr: an expression to check
        :return: True, iff expr is constant
        """
        self.smt_solver.reset()

        if not isinstance(expr, expression.ColumnRef):
            return False

        def get_value():
            model = self.smt_solver.model()
            value = None
            for decl in model.decls():
                if str(expr) == decl.name():
                    value = model[decl]

            return value

        # get some value for the expression
        separator = "\n"

        program = separator.join(self.declarations)
        program += separator
        program += separator.join(self.assertions)

        self.smt_solver.from_string(program)

        if not self.smt_solver.check() == z3.sat:
            return False

        first_value = get_value()
        if first_value is None:
            return False

        # check that there can't be a different value
        program += separator
        program += f"(assert (not (= {expr} {first_value})))"

        self.smt_solver.from_string(program)

        if self.smt_solver.check() == z3.unsat:
            return True

        second_value = get_value()
        if second_value is None:
            return False

        return first_value == second_value

    def values_expression_range(self, expr):
        """Finds a set of values that an expression can take

        :param expr: an expression for which a set of values is searched for
        :return: a set of values
        """
        separator = "\n"

        # try to deduce expr type
        supported_types = ["Int", "Real", "String", "Bool"]
        actual_type = None
        for possible_type in supported_types:
            program = separator.join(
                self.declarations
                | {f"(declare-const expr_value {possible_type})"}
            )
            program += separator
            program += separator.join(
                self.assertions | {f"(assert (= expr_value {expr}))"}
            )

            try:
                self.smt_solver.from_string(program)
            except z3.z3types.Z3Exception:
                self.smt_solver.reset()
                continue

            actual_type = possible_type

        if actual_type is None:
            raise ValueError("unsupported expression type")

        self.smt_solver.reset()

        # get expression range
        program = separator.join(
            self.declarations | {f"(declare-const expr_value {actual_type})"}
        )
        program += separator
        program += separator.join(
            self.assertions | {f"(assert (= expr_value {expr}))"}
        )

        local_assertions = set()
        expr_range = set()

        while self.smt_solver.check() == z3.sat:
            current_model = self.smt_solver.model()
            for decl in current_model.decls():
                if decl.name() == "expr_value":
                    value = current_model[decl]
                    value_str = value.sexpr()
                    expr_range.add(value)
                    local_assertions.add(
                        f"(assert (not (= expr_value {value_str})))"
                    )

            self.smt_solver.from_string(
                program + separator + separator.join(local_assertions)
            )

        return expr_range
