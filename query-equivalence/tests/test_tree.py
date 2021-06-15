import pytest

from calcite import CASES
from equivalence import db, tree


@pytest.mark.parametrize("case", CASES, ids=str)
def test_tree_from_dict(case):
    from pprint import pprint

    plan_generator = db.PlanGenerator(
        db.ConnectionParameters("postgres", "postgres", "demo")
    )

    plan1 = plan_generator.get_json(case.query1)
    pprint(plan1)
    tree1 = tree.from_dict(plan1)
    pprint(tree1)

    plan2 = plan_generator.get_json(case.query2)
    pprint(plan2)
    tree2 = tree.from_dict(plan2)
    pprint(tree2)
