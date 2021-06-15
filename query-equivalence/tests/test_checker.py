import pytest

from equivalence import db, checker
from tests.calcite import CASES


@pytest.fixture
def checker_fixture():
    plan_generator = db.PlanGenerator(
        db.ConnectionParameters("postgres", "postgres", "demo")
    )

    return checker.Checker(plan_generator)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_calcite(case, checker_fixture):
    # skip = [
    #     "testAddRedundantSemiJoinRule",
    #     "testAggregateConstantKeyRule3",
    #     "testCastInAggregateExpandDistinctAggregatesRule",
    #     "testDecorrelateTwoExists",
    #     "testDecorrelateTwoIn",
    # ]
    #
    # if case.name in skip:
    #     return

    skip = [
        # "testAddRedundantSemiJoinRule",
        # "testCastInAggregateExpandDistinctAggregatesRule"
    ]
    if case.name in skip:
        return

    assert checker_fixture.check(case.query1, case.query2)
