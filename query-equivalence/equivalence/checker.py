import copy

from equivalence import solver, tree, expression, db


class Checker:
    def __init__(self, plan_generator: db.PlanGenerator):
        self.plan_generator = plan_generator
        self.solver = solver.Solver(plan_generator.get_schema_info())

    def check(self, query1: str, query2: str):
        """Checks whether the two queries are equivalent

        :param query1: first query
        :param query2: second query
        :return: True, iff query1 is equivalent to the query2
        """
        plan_dict1 = self.plan_generator.get_json(query1)
        plan_dict2 = self.plan_generator.get_json(query2)

        for relation, rows in self.plan_generator.values_relations.items():
            self.solver.add_values_context(relation, rows)

        tree1 = tree.from_dict(plan_dict1)
        tree2 = tree.from_dict(plan_dict2)

        if len(tree1.output) != len(tree2.output):
            return False

        # normalize with solver-free rules
        normalized_tree1 = tree.normalize(tree1)
        normalized_tree2 = tree.normalize(tree2)

        # normalize with solver-aided rules
        normalized_tree1 = self.eliminate_redundant_joins(normalized_tree1)
        normalized_tree2 = self.eliminate_redundant_joins(normalized_tree2)

        # normalized_tree1.to_graph().render("1")
        # normalized_tree2.to_graph().render("2")

        return self.compare_trees(normalized_tree1, normalized_tree2, False)

    def eliminate_constants(self, exprs):
        """Removes constant expressions from a set by checking constancy with a solver

        :param exprs: set of expressions
        :return: set of non-constant expressions
        """
        result = set()
        for s in exprs:
            expr = expression.from_str(s)
            if not self.solver.is_constant(expr):
                result.add(s)

        return result

    def eliminate_redundant_joins(self, root):
        """Removes redundant scans in a join by checking primary key equality

        :param root: query tree's root
        :return: query tree without redundant joins
        """
        tree_copy = copy.deepcopy(root)

        def traverse(node):
            traversed_children = [traverse(child) for child in node.children]

            if isinstance(node, tree.Scan):
                self.solver.add_alias(node.relation, node.alias)

            if isinstance(node, tree.Filter):
                predicate_expr = expression.from_str(node.predicate)

                self.solver.compare_expressions(predicate_expr, predicate_expr)
                self.solver.add_assertion_str(f"(assert {predicate_expr})")

                referenced_relations = [
                    relation
                    for expr in node.output
                    for relation in expression.extract_referenced_relations(
                        expression.from_str(expr)
                    )
                ]

                child = node.children[0]
                if isinstance(child, tree.Product):
                    factors = child.children

                    if len(factors) <= 1:
                        raise ValueError(
                            f"invalid children count for product: {len(factors)}"
                        )

                    result = []
                    all_scans = all(
                        [isinstance(factor, tree.Scan) for factor in factors]
                    )

                    if all_scans:
                        if len(factors) == 2:
                            node_copy = copy.deepcopy(node)
                            node_copy.children = traversed_children
                            return node_copy

                        for scan1, scan2 in zip(factors, factors[1:]):
                            if scan1.relation != scan2.relation:
                                result.append(scan1)
                                continue

                            scan1_referenced = (
                                scan1.alias in referenced_relations
                            )
                            scan2_referenced = (
                                scan2.alias in referenced_relations
                            )

                            if scan1_referenced and scan2_referenced:
                                continue

                            scan1_primary_key = self.solver.schema_info[
                                scan1.relation
                            ]["PRIMARY KEY"]
                            scan2_primary_key = self.solver.schema_info[
                                scan1.relation
                            ]["PRIMARY KEY"]

                            if self.solver.check_validity(
                                expression.Equal(
                                    expression.ColumnRef(
                                        scan1_primary_key.column_name,
                                        scan1.alias,
                                    ),
                                    expression.ColumnRef(
                                        scan2_primary_key.column_name,
                                        scan2.alias,
                                    ),
                                )
                            ):
                                result.append(scan1)

                        child.children = result

                        return node

            node_copy = copy.deepcopy(node)
            node_copy.children = traversed_children
            return node_copy

        return traverse(tree_copy)

    def compare_trees(self, tree1, tree2, skip_types_mismatch=True):
        """Compares two normalized query trees

        :param tree1: first tree
        :param tree2: second tree
        :param skip_types_mismatch: do not raise an exception if trees structure is different
        :return: True, iff first tree is same as the second one
        """

        def isinstance_both_local(t):
            return isinstance_both(tree1, tree2, t)

        if type(tree1) != type(tree2):
            if skip_types_mismatch:
                return False

            type1_name = type(tree1).__name__
            type2_name = type(tree2).__name__

            raise ValueError(
                f"node types mismatch: {type1_name} and {type2_name}"
            )

        if len(tree1.children) != len(tree2.children):
            return False

        children_are_equivalent = True
        for child1, child2 in zip(tree1.children, tree2.children):
            if not self.compare_trees(child1, child2):
                children_are_equivalent = False

        # print(f"comparing {type(tree1).__name__} with {type(tree2).__name__}")

        if isinstance_both_local(tree.Scan):
            self.solver.add_alias(tree1.relation, tree1.alias)
            self.solver.add_alias(tree2.relation, tree2.alias)

            tree1_values_relation = tree1.relation.startswith("qe_values")
            tree2_values_relation = tree2.relation.startswith("qe_values")

            if tree1_values_relation and tree2_values_relation:
                for column1, column2 in zip(tree1.output, tree2.output):
                    if column1 == column2:
                        continue

                    expr1 = expression.from_str(column1)
                    expr2 = expression.from_str(column2)

                    range1 = self.solver.values_expression_range(expr1)
                    range2 = self.solver.values_expression_range(expr2)

                    if range1 != range2:
                        return False

                    self.solver.add_assertion_str(
                        f"(assert (= {expr1} {expr2}))"
                    )

                return True

            return tree1.relation == tree2.relation

        if isinstance_both_local(tree.Result):
            for column1, column2 in zip(tree1.output, tree2.output):
                if column1 == column2:
                    continue

                if not self.solver.compare_expression_strings(
                    column1, column2
                ):
                    return False

            return True

        if isinstance_both_local(tree.Filter):
            expr1 = expression.from_str(tree1.predicate)
            expr2 = expression.from_str(tree2.predicate)

            if not self.solver.compare_expressions(expr1, expr2):
                return False

            self.solver.add_assertion_str(f"(assert {expr1})")

            return True

        if not children_are_equivalent:
            return False

        if isinstance_both_local(tree.Aggregate):
            key1 = self.eliminate_constants(tree1.key)
            key2 = self.eliminate_constants(tree2.key)

            key1_list = sorted(list(key1))
            key2_list = sorted(list(key2))

            for s1, s2 in zip(key1_list, key2_list):
                if not self.solver.compare_expression_strings(s1, s2):
                    return False

            return True

        if isinstance_both_local(tree.Sort):
            if len(tree1.key) != len(tree2.key):
                return False

            for s1, s2 in zip(tree1.key, tree2.key):
                if not self.solver.compare_expression_strings(s1, s2):
                    return False

            return True

        if isinstance_both_local(tree.Limit):
            return tree1.rows_count == tree2.rows_count

        if isinstance_both_local(
            (
                tree.Product,
                tree.Union,
                tree.UnionAll,
                tree.Intersect,
                tree.IntersectAll,
                tree.Except,
                tree.ExceptAll,
                tree.LeftJoin,
                tree.RightJoin,
            )
        ):
            return True

        if isinstance_both_local(tree.Window):
            raise ValueError("Window")

        raise NotImplementedError(
            f"unsupported node type: {type(tree).__name__}"
        )


def isinstance_both(tree1, tree2, node_type):
    """Checks if two nodes have the same given type

    :param tree1: first node
    :param tree2: second node
    :param node_type: type to check
    :return: True, iff both trees are of the node_type
    """
    isinstance_tree1 = isinstance(tree1, node_type)
    isinstance_tree2 = isinstance(tree2, node_type)

    return isinstance_tree1 and isinstance_tree2
