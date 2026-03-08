from collections import deque


class HierarchyNode:
    def __init__(self, depth=0, token=None):
        self.depth = depth
        self.token = token
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def __repr__(self):
        if self.token:
            return f"Node(depth={self.depth}, token={self.token['token']})"
        return f"Node(depth={self.depth}, children={len(self.children)})"


def min_indent_in_range(tokens, s, e):
    return min(tokens[i]["indent_level"] for i in range(s, e + 1))


def all_same_indent(tokens, s, e, indent):
    for i in range(s, e + 1):
        if tokens[i]["indent_level"] != indent:
            return False
    return True


def build_segments(tokens, boundary_indices):
    boundary_set = set(boundary_indices)

    segments = []
    current = []

    for i, token in enumerate(tokens):

        if i in boundary_set and current:
            segments.append(current)
            current = []

        current.append(token)

    if current:
        segments.append(current)

    return segments


def build_hierarchy(tokens):
    """
    Build hierarchy tree from token list using indent_level.
    tokens: list of token dicts containing 'indent_level'
    """

    root = HierarchyNode(depth=0)

    if not tokens:
        return root

    queue = deque()
    queue.append((root, 0, len(tokens) - 1))

    while queue:
        node, s, e = queue.popleft()

        if s == e:
            node.add_child(HierarchyNode(depth=node.depth + 1, token=tokens[s]))
            continue

        min_indent = min_indent_in_range(tokens, s, e)
        same = all_same_indent(tokens, s, e, min_indent)

        if same:
            for i in range(s, e + 1):
                node.add_child(HierarchyNode(depth=node.depth + 1, token=tokens[i]))
            continue

        i = s
        while i <= e:
            if tokens[i]["indent_level"] == min_indent:
                node.add_child(HierarchyNode(depth=node.depth + 1, token=tokens[i]))
                i += 1
            else:
                group_start = i

                while i <= e and tokens[i]["indent_level"] > min_indent:
                    i += 1

                group_end = i - 1

                internal = HierarchyNode(depth=node.depth + 1)
                node.add_child(internal)

                queue.append((internal, group_start, group_end))

    return root