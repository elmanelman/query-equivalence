import pytest

from calcite import CASES
from equivalence import expression, db, tree


def extract_expressions(plan):
    result = []

    def traverse(node):
        nonlocal result

        result += node.output

        if isinstance(node, (tree.Filter, tree.LeftJoin, tree.RightJoin)):
            result.append(node.predicate)

        if isinstance(node, tree.Aggregate):
            if node.key is not None:
                result += node.key

        if isinstance(node, tree.Sort):
            result += node.key

        for child in node.children:
            traverse(child)

    traverse(plan)

    result = list(filter(lambda e: e is not None, result))

    return result


@pytest.mark.parametrize("case", CASES, ids=str)
def test_expression_from_str(case):
    plan_generator = db.PlanGenerator(
        db.ConnectionParameters("postgres", "postgres", "demo")
    )

    plan1 = plan_generator.get_json(case.query1)
    tree1 = tree.from_dict(plan1)
    plan1_expressions = extract_expressions(tree1)
    for s in plan1_expressions:
        expression.from_str(s)

    plan2 = plan_generator.get_json(case.query2)
    tree2 = tree.from_dict(plan2)
    plan2_expressions = extract_expressions(tree2)
    for s in plan2_expressions:
        expression.from_str(s)

    with open("all_expressions.txt", "a") as f:
        all_map = map(lambda s: s + "\n", set(plan1_expressions) | set(plan2_expressions))
        all_list_sorted = sorted(list(all_map), key=lambda s: len(s), reverse=True)
        f.writelines(all_list_sorted)
