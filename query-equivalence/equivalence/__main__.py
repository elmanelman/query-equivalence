from pprint import pprint

from equivalence import values, db, tree


def main():
    # query = "SELECT * FROM (SELECT * FROM (VALUES  (10, 1),  (30, 3)) AS t UNION ALL SELECT * FROM (VALUES  (20, 2)) AS t0) AS t1 WHERE t1.COLUMN1 + t1.COLUMN2 > 30"
    # plan_generator = db.PlanGenerator(
    #     db.ConnectionParameters("postgres", "postgres", "demo")
    # )
    #
    # plan_dict = plan_generator.get_json(query)
    # for relation, rows in plan_generator.values_relations.items():
    #     print(values.solver_context(relation, rows))

    plan_generator = db.PlanGenerator(
        db.ConnectionParameters("postgres", "postgres", "demo")
    )
    # q = "SELECT 1 FROM (SELECT EMP.DEPTNO FROM EMP AS EMP WHERE EMP.DEPTNO > 7 UNION ALL SELECT EMP0.DEPTNO FROM EMP AS EMP0 WHERE EMP0.DEPTNO > 10) AS t3 INNER JOIN EMP AS EMP1 ON t3.DEPTNO = EMP1.DEPTNO"
    # q = "SELECT 1 FROM (SELECT EMP2.DEPTNO FROM EMP AS EMP2 WHERE EMP2.DEPTNO > 7 UNION ALL SELECT EMP3.DEPTNO FROM EMP AS EMP3 WHERE EMP3.DEPTNO > 10) AS t9 INNER JOIN (SELECT * FROM EMP AS EMP4 WHERE EMP4.DEPTNO > 7 OR EMP4.DEPTNO > 10) AS t10 ON t9.DEPTNO = t10.DEPTNO"
    # q = "SELECT 1 FROM EMP AS EMP INNER JOIN DEPT AS DEPT ON EMP.DEPTNO = DEPT.DEPTNO"
    # q = "SELECT 1 FROM EMP AS EMP0 INNER JOIN DEPT AS DEPT0 ON EMP0.DEPTNO = DEPT0.DEPTNO INNER JOIN DEPT AS DEPT1 ON EMP0.DEPTNO = DEPT1.DEPTNO"

    # q = "SELECT COUNT(*) AS C FROM EMP AS EMP WHERE EMP.DEPTNO = 10 GROUP BY EMP.DEPTNO, EMP.SAL"

    q = """
select t.flight_id from (
    select f.flight_id, f.status, tf.amount from (
        select * from bookings.flights f order by status
    ) f left join (
        select * from bookings.ticket_flights tf
    ) tf on f.flight_id = tf.flight_id order by f.status desc
) t order by t.flight_id;
    """

    plan_dict = plan_generator.get_json(q)
    pprint(plan_dict)
    plan_tree = tree.from_dict(plan_dict)
    plan_tree = tree.normalize(plan_tree)
    plan_tree.to_graph().view()

    # from equivalence import expression

    # print(expression.from_str("2.0 * flight_id / 3.0 > 1000.0 and not upper(status) = 'Arrived'"))
    # print(expression.from_str("1 + NULL"))
    # print(expression.simplify_nulls(expression.from_str("case null when 1 then 1 else 2 end")))

    # print(
    #     expression.extract_referenced_relations(
    #         expression.from_str("1 + 2 + emp0.deptno")
    #     )
    # )


if __name__ == "__main__":
    main()
