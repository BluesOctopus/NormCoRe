"""Edit applicator for RECODE spy-grammar Pass A⁻¹."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Edit:
    node: object
    action: str = "insert"
    content: str = ""
    priority: float = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edit):
            return NotImplemented
        return self.node == other.node and self.content == other.content


@dataclass
class FakeNode:
    start_byte: int
    end_byte: int
    text: bytes | str


def traverse_all_children(node, results: list) -> None:
    results.append(node)
    for n in node.children:
        traverse_all_children(n, results)


def traverse_type(node, results: list, kind: str) -> None:
    if node.type == kind:
        results.append(node)
    for n in node.children:
        traverse_type(n, results, kind)


def search_edits(nodes, action: str, target_type: str, content: str) -> list[Edit]:
    return [
        Edit(node=node, action=action, content=content)
        for node in nodes
        if node.type == target_type
    ]


def apply_spy_edits(code: str, edits: list[Edit], *, indent_spaces: int = 4) -> str:
    """Apply SimPy-style spy→py edits (insert/replace/append/indent/dedent/newline)."""
    bcode = bytes(code, "utf8")
    remove: set[int] = set()
    for edit in edits:
        if edit.action == "cancel":
            for i, other in enumerate(edits):
                if edit == other:
                    remove.add(i)
    edits = [e for i, e in enumerate(edits) if i not in remove]

    def sort_key(x: Edit):
        if x.action == "append":
            return x.node.end_byte - x.priority
        if x.action == "insert":
            return x.node.start_byte + 0.1 - x.priority
        if x.action == "replace":
            return x.node.start_byte + 0.3 - x.priority
        if x.action == "mask":
            return x.node.start_byte + 0.2 - x.priority
        if x.action == "indent":
            return x.node.start_byte + 0.25 - x.priority
        if x.action == "dedent":
            return x.node.start_byte + 0.06 - x.priority
        if x.action == "newline":
            return x.node.start_byte + 0.24 - x.priority
        return float(x.node.start_byte)

    edits = sorted(edits, key=sort_key)
    margin = 0
    indent = 0
    for edit in edits:
        start_byte = edit.node.start_byte + margin
        end_byte = edit.node.end_byte + margin
        if edit.action == "insert":
            content = bytes(edit.content, "utf8")
            bcode = bcode[:start_byte] + content + bcode[start_byte:]
            margin += len(content)
        elif edit.action == "replace":
            content = bytes(edit.content, "utf8")
            bcode = bcode[:start_byte] + content + bcode[end_byte:]
            margin += len(content) - (end_byte - start_byte)
        elif edit.action == "append":
            content = bytes(edit.content, "utf8")
            bcode = bcode[:end_byte] + content + bcode[end_byte:]
            margin += len(content)
        elif edit.action == "indent":
            indent += indent_spaces
            content = bytes("\n" + " " * indent, "utf8")
            bcode = bcode[:start_byte] + content + bcode[start_byte:]
            margin += indent + 1
        elif edit.action == "dedent":
            indent -= indent_spaces
        elif edit.action == "newline":
            content = bytes("\n" + " " * indent, "utf8")
            bcode = bcode[:start_byte] + content + bcode[start_byte:]
            margin += indent + 1
    return bcode.decode("utf8")
