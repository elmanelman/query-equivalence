from __future__ import annotations

import copy
from abc import ABCMeta, abstractmethod
from typing import Optional
from uuid import uuid4, UUID

import graphviz

Expression = str


class Node(metaclass=ABCMeta):
    def __init__(
        self, output: list[Expression], children: Optional[list[Node]] = None
    ) -> None:
        self.output = output
        self.children = [] if children is None else children

    def compare_structure(self, other: Node) -> bool:
        """Compare nodes by recursively comparing only their types

        :param other: node to compare with
        :return: comparison result
        """
        if type(self) is not type(other):
            return False

        if len(self.children) != len(other.children):
            return False

        for self_child, other_child in zip(self.children, other.children):
            if not self_child.compare_structure(other_child):
                return False

        return True

    @abstractmethod
    def label(self) -> str:
        """Returns node label for tree visualization

        :return: node label
        """
        return ""

    @staticmethod
    def attributes() -> dict[str, str]:
        """Returns node attributes for tree visualization

        :return: node attributes dictionary
        """
        return {}

    def to_graph(self) -> graphviz.Digraph:
        """Build a graphviz directed graph of a given node

        :return: graph of a given node
        """
        default_node_attr = {"shape": "box"}
        default_edge_attr = {"arrowhead": "none"}

        graph = graphviz.Digraph(
            node_attr=default_node_attr, edge_attr=default_edge_attr
        )

        def traverse(node: Node, uuid: UUID) -> None:
            graph.node(str(uuid), node.label(), _attributes=node.attributes())

            for child in node.children:
                child_uuid = uuid4()
                traverse(child, child_uuid)
                graph.edge(str(uuid), str(child_uuid))

        traverse(self, uuid4())

        return graph


class Scan(Node):
    """Scan all rows from the relation"""

    def __init__(
        self,
        output: list[Expression],
        relation: str,
        alias: Optional[str] = None,
    ) -> None:
        super().__init__(output)

        self.relation = relation
        self.alias = alias

    def label(self) -> str:
        if self.alias is None:
            return self.relation

        return f"{self.relation} as {self.alias}"


class Result(Node):
    """Return a row of constant expressions"""

    def __init__(self, output: list[Expression]) -> None:
        super().__init__(output)

    def label(self) -> str:
        return ", ".join(self.output)


class Filter(Node):
    """Filter rows from a child node by predicate"""

    def __init__(
        self, output: list[Expression], child: Node, predicate: Expression
    ) -> None:
        super().__init__(output, [child])

        self.predicate = predicate

    def label(self) -> str:
        return self.predicate


class Aggregate(Node):
    """Aggregate rows from a child node, possibly computing some aggregate functions"""

    def __init__(
        self, output: list[Expression], child: Node, key: set[Expression]
    ) -> None:
        super().__init__(output, [child])

        self.key = key

    def label(self) -> str:
        return f"group by {', '.join(self.key)}\n{self.output}"


class Operator:
    @staticmethod
    def attributes():
        return {"shape": "circle"}


class Product(Operator, Node):
    """Cartesian product"""

    def __init__(self, output: list[Expression], children: list[Node]) -> None:
        super().__init__(output, children)

    def label(self) -> str:
        return "×"


class Union(Operator, Node):
    """Set union of some subqueries"""

    def __init__(self, output: list[Expression], children: list[Node]) -> None:
        super().__init__(output, children)

    def label(self) -> str:
        return "∪"


class UnionAll(Operator, Node):
    """Bag (multiset) union of some subqueries"""

    def __init__(self, output: list[Expression], children: list[Node]) -> None:
        super().__init__(output, children)

    def label(self) -> str:
        return "∪ all"


class Intersect(Operator, Node):
    """Set intersection of some subqueries"""

    def __init__(self, output: list[Expression], children: list[Node]) -> None:
        super().__init__(output, children)

    def label(self) -> str:
        return "∩"


class IntersectAll(Operator, Node):
    """Bag (multiset) intersection of some subqueries"""

    def __init__(self, output: list[Expression], children: list[Node]) -> None:
        super().__init__(output, children)

    def label(self) -> str:
        return "∩ all"


class Except(Operator, Node):
    """Set difference of two subqueries"""

    def __init__(self, output: list[Expression], children: list[Node]) -> None:
        super().__init__(output, children)

    def label(self) -> str:
        return "∖"


class ExceptAll(Operator, Node):
    """Bag (multiset) difference of two subqueries"""

    def __init__(self, output: list[Expression], children: list[Node]) -> None:
        super().__init__(output, children)

    def label(self) -> str:
        return "∖ all"


class Sort(Node):
    """Sort rows from a child node"""

    def __init__(
        self,
        output: list[Expression],
        child: Node,
        key: list[Expression],
    ) -> None:
        super().__init__(output, [child])

        self.key = key

    def label(self) -> str:
        return f"order by {', '.join(self.key)}"


class Limit(Node):
    """Take a certain number of rows from a child node"""

    def __init__(
        self,
        output: list[Expression],
        child: Node,
        rows_count: int,
    ) -> None:
        super().__init__(output, [child])

        self.rows_count = rows_count

    def label(self) -> str:
        return f"limit {self.rows_count} rows"


class LeftJoin(Node):
    """Left join of two subqueries"""

    def __init__(
        self,
        output: list[Expression],
        children: list[Node],
        predicate: Expression,
    ) -> None:
        super().__init__(output, children)

        self.predicate = predicate

    def label(self) -> str:
        return f"left join on {self.predicate}"


class RightJoin(Node):
    """Right join of two subqueries"""

    def __init__(
        self,
        output: list[Expression],
        children: list[Node],
        predicate: Expression,
    ) -> None:
        super().__init__(output, children)

        self.predicate = predicate

    def label(self) -> str:
        return f"right join on {self.predicate}"


class Window(Node):
    """Window function"""

    def __init__(self, output: list[Expression], child: Node) -> None:
        super().__init__(output, [child])

    def label(self) -> str:
        return "window"


def from_dict(plan_dict) -> Node:
    """Builds a query tree from a PostgreSQL plan

    :param plan_dict: plan dictionary
    :return: query tree
    """

    def get_children(node_dict):
        return node_dict["Plans"]

    def get_child(node_dict):
        children = get_children(node_dict)

        if len(children) != 1:
            raise ValueError("get_child supports only unary nodes")

        return children[0]

    def get_type(node_dict):
        return node_dict["Node Type"]

    def get_output(node_dict):
        return node_dict.get("Output", [])

    def traverse(node_dict) -> Node:
        node_type = get_type(node_dict)

        output = get_output(node_dict)

        if node_type == "Seq Scan":
            relation = node_dict["Relation Name"]
            alias = node_dict["Alias"]

            scan_node = Scan(output, relation, alias)

            if "Filter" in node_dict:
                return Filter(output, scan_node, node_dict["Filter"])

            return scan_node

        if node_type == "Subquery Scan":
            return traverse(get_child(node_dict))

        if node_type == "Result":
            if "One-Time Filter" in node_dict:
                return Filter(
                    output, Result(output), node_dict["One-Time Filter"]
                )

            return Result(output)

        if node_type in ("Nested Loop", "Merge Join"):
            children = [traverse(child) for child in get_children(node_dict)]

            join_type = node_dict["Join Type"]

            if join_type == "Inner":
                product_node = Product(output, children)

                if "Join Filter" in node_dict:
                    predicate = node_dict["Join Filter"]

                    return Filter(output, product_node, predicate)

                return product_node

            predicate = node_dict.get("Join Filter")

            if join_type in ("Left", "Anti", "Semi"):
                return LeftJoin(output, children, predicate)

            if join_type == "Right":
                return RightJoin(output, children, predicate)

            if join_type == "Full":
                return Union(
                    output,
                    [
                        LeftJoin(output, children, predicate),
                        RightJoin(output, children, predicate),
                    ],
                )

            raise ValueError(f"unsupported join type: {join_type}")

        if node_type == "Hash Join":
            join_type = node_dict["Join Type"]

            if join_type == "Full":
                children = [
                    traverse(child) for child in get_children(node_dict)
                ]
                predicate = node_dict.get("Hash Cond")

                return Union(
                    output,
                    [
                        LeftJoin(output, children, predicate),
                        RightJoin(output, children, predicate),
                    ],
                )

            raise ValueError(f"unsupported hash join type: {join_type}")

        if node_type == "Hash":
            return traverse(get_child(node_dict))

        if node_type in ("Aggregate", "Group"):
            key = node_dict.get("Group Key", [])
            child = traverse(get_child(node_dict))

            return Aggregate(output, child, set(key))

        if node_type in ("Append", "Merge Append"):
            children = [traverse(child) for child in get_children(node_dict)]

            if output is None:
                output = []

            return UnionAll(output, children)

        if node_type == "Unique":
            child_type = get_type(get_child(node_dict))
            if child_type == "Sort":
                child_child_type = get_type(get_child(get_child(node_dict)))
                if child_child_type == "Append":
                    children = [
                        traverse(child) for child in get_children(node_dict)
                    ]

                    return Union(output, children)

        if node_type == "SetOp":
            children = [traverse(child) for child in get_children(node_dict)]

            operator_name = node_dict["Command"]
            operator_map = {
                "Intersect": Intersect,
                "Intersect All": IntersectAll,
                "Except": Except,
                "Except All": ExceptAll,
            }

            if operator_name in operator_map:
                return operator_map[operator_name](output, children)

            raise ValueError("unknown set operator: " + operator_name)

        if node_type == "WindowAgg":
            child = traverse(get_child(node_dict))

            return Window(output, child)

        if node_type == "Sort":
            key = node_dict["Sort Key"]
            child = traverse(get_child(node_dict))

            return Sort(output, child, key)

        if node_type == "Limit":
            child = traverse(get_child(node_dict))
            rows_count = node_dict["Plan Rows"]

            return Limit(output, child, rows_count)

        raise ValueError(f"unknown plan node type: {node_type}")

    node = traverse(plan_dict)

    return node


def is_filter(node):
    return isinstance(node, Filter)


def is_product(node):
    return isinstance(node, Product)


def is_aggregate(node):
    return isinstance(node, Aggregate)


def is_sorting(node):
    return isinstance(node, Sort)


def pull_filters(tree):
    """Pull filters in a query tree by merging predicates with conjunction

    :param tree: query tree
    :return: normalized query tree
    """
    supported_operators = (Product, Sort, LeftJoin, RightJoin)

    def traverse(node):
        traversed_children = [traverse(child) for child in node.children]

        if isinstance(node, supported_operators):
            filters = []
            result_children = []

            for i, child in enumerate(traversed_children):
                if is_filter(child):
                    filters.append(child.predicate)
                    result_children.append(child.children[0])
                else:
                    result_children.append(child)

            if len(filters) == 0:
                return node

            node_copy = copy.deepcopy(node)
            node_copy.children = result_children

            return Filter(node.output, node_copy, " AND ".join(filters))
        elif is_filter(node) and is_filter(traversed_children[0]):
            return Filter(
                node.output,
                traversed_children[0].children[0],
                f"{node.predicate} AND {traversed_children[0].predicate}",
            )
        else:
            node_copy = copy.deepcopy(node)
            node_copy.children = traversed_children
            return node_copy

    traversed = traverse(tree)

    return traversed


def redundant_sorts(tree):
    """Removes redundant sort nodes

    :param tree: query tree
    :return: normalized query tree
    """
    supported_operators = (Product, LeftJoin, RightJoin, Aggregate)

    def traverse(node):
        traversed_children = [traverse(child) for child in node.children]

        if isinstance(node, supported_operators):
            result_children = []

            for i, child in enumerate(traversed_children):
                if is_sorting(child):
                    result_children.append(child.children[0])
                else:
                    result_children.append(child)

            node_copy = copy.deepcopy(node)
            node_copy.children = result_children

            return node_copy

        if is_sorting(node) and is_sorting(traversed_children[0]):
            node_copy = copy.deepcopy(node)
            node_copy.children = traversed_children[0].children

            return node_copy

        node_copy = copy.deepcopy(node)
        node_copy.children = traversed_children
        return node_copy

    traversed = traverse(tree)

    return traversed


def merge_products(tree):
    """Merges product nodes to reduce tree's depth

    :param tree: query tree
    :return: normalized query tree
    """

    def traverse(node):
        traversed_children = [traverse(child) for child in node.children]

        if is_product(node):
            result_children = []

            for child in traversed_children:
                if is_product(child):
                    result_children += child.children
                else:
                    result_children.append(child)

            node_copy = copy.deepcopy(node)
            node_copy.children = result_children

            return node_copy

        node_copy = copy.deepcopy(node)
        node_copy.children = traversed_children
        return node_copy

    traversed = traverse(tree)

    return traversed


def aggregate_reduction(tree):
    """Removes nested aggregations by checking if upper key is a subset of a lower one

    :param tree: query tree
    :return: normalized query tree
    """

    def traverse(node):
        if len(node.children) == 0:
            return node

        traversed_children = [traverse(child) for child in node.children]

        if is_aggregate(node) and is_aggregate(traversed_children[0]):
            if node.key.issubset(traversed_children[0].key):
                node_copy = copy.deepcopy(node)
                node_copy.children = traversed_children[0].children

                return node_copy

        node_copy = copy.deepcopy(node)
        node_copy.children = traversed_children
        return node_copy

    traversed = traverse(tree)

    return traversed


def normalize(tree):
    """Normalizes query tree by applying various equivalent transformations

    :param tree: query tree
    :return: normalized query tree
    """
    sequence = (
        redundant_sorts,
        pull_filters,
        aggregate_reduction,
        merge_products,
    )

    tree_copy = copy.deepcopy(tree)
    for algorithm in sequence:
        tree_copy = algorithm(tree_copy)

    return tree_copy
