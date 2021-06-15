import secrets

import pglast
import pglast.printers
from pglast.printer import RawStream


def extract_nodes(parse_tree):
    """Extracts values nodes from a PostgreSQL query AST

    :param parse_tree: abstract syntax tree of a query
    :return: list of extracted values nodes
    """
    nodes = []

    def traverse(node):
        if isinstance(node, dict):
            for key in node:
                if type(node[key]) is dict and "valuesLists" in node[key]:
                    nodes.append(node)

                traverse(node[key])

        if isinstance(node, list):
            for child in node:
                traverse(child)

    traverse(parse_tree)

    return nodes


def node_to_sql(node):
    """Prints an arbitrary AST node

    :param node: node to print
    :return: printed node
    """
    return RawStream()(pglast.Node({"RawStmt": {"stmt": node}}))


def replace_with_relation(node, relation: str):
    """Substitutes a relation reference in the values node

    :param node: values node
    :param relation: relation name to substitute
    """
    node["SelectStmt"]["fromClause"] = [
        {
            "RangeVar": {
                "inh": True,
                "relname": relation,
                "relpersistence": "p",
            }
        }
    ]

    node["SelectStmt"]["targetList"] = [
        {"ResTarget": {"val": {"ColumnRef": {"fields": [{"A_Star": {}}]}}}}
    ]

    node["SelectStmt"].pop("valuesLists", None)


def prepare_query(query: str):
    """Extracts all values nodes from a query and replaces them with temporary relations

    :param query: input query
    :return: query with substituted temporary relations
    """
    parse_tree = pglast.parse_sql(query)
    values_nodes = extract_nodes(parse_tree)

    if len(values_nodes) == 0:
        return query, {}

    temporary_relations = {}

    for node in values_nodes:
        relation = f"qe_values_{secrets.token_hex(2)}"

        temporary_relations[relation] = node_to_sql(node)
        replace_with_relation(node, relation)

    result_query = RawStream()(pglast.Node(parse_tree))

    return result_query, temporary_relations


def solver_context(relation, rows):
    """Builds a solver context (declarations and assertions) from a values' relation definition

    :param relation: relation name
    :param rows: relation rows
    :return: declaration and assertions
    """
    declarations = set()
    type_map = {int: "Int", str: "String", bool: "Bool"}
    for i, value in enumerate(rows[0]):
        declarations.add(
            f"(declare-const {relation}_column{i + 1} {type_map[type(value)]})"
        )

    assertions = set()
    for row in rows:
        row_assertions = []
        for i, value in enumerate(row):
            value_str = (
                str(value).lower() if isinstance(value, bool) else str(value)
            )

            row_assertions.append(f"(= {relation}_column{i + 1} {value_str})")

        assertions.add(f"(and {' '.join(row_assertions)})")

    if len(assertions) == 1:
        return declarations, {f"(assert {assertions.pop()})"}

    return declarations, {f"(assert (or {' '.join(assertions)}))"}
