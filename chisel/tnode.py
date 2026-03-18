"""
TNode - Mutable Tree Node for Dependency Transformations

Lightweight parallel structure to spaCy Doc that allows tree transformations.
Decouples from spaCy's immutable Doc and makes transformations testable.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from spacy.tokens import Doc, Token


@dataclass
class TNode:
    """Mutable tree node for dependency tree transformations."""
    index: int
    word: str
    lemma: str
    pos: str
    tag: str
    dep: str
    head_index: int
    children: List['TNode'] = field(default_factory=list)

    # Custom attributes for transformations
    is_extracted: bool = False
    has_bind: bool = False
    controller_index: Optional[int] = None
    corrected_dep: Optional[str] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def get_dep(self) -> str:
        """Get dependency label (corrected if available)."""
        return self.corrected_dep if self.corrected_dep else self.dep

    def add_child(self, child: 'TNode'):
        """Add child node."""
        if child not in self.children:
            self.children.append(child)

    def remove_child(self, child: 'TNode'):
        """Remove child node."""
        if child in self.children:
            self.children.remove(child)

    def get_children_by_dep(self, dep: str) -> List['TNode']:
        """Get all children with given dependency."""
        return [c for c in self.children if c.get_dep() == dep]

    def has_dep(self, dep: str) -> bool:
        """Check if has child with given dependency."""
        return any(c.get_dep() == dep for c in self.children)

    def __repr__(self) -> str:
        dep = self.get_dep()
        return f"TNode({self.index}:{self.word}/{self.pos}/{dep})"


def doc_to_tnodes(doc: Doc) -> List[TNode]:
    """
    Convert spaCy Doc to list of TNodes.

    Args:
        doc: spaCy Doc

    Returns:
        List of TNodes (index 0 = ROOT, index 1+ = tokens)
    """
    # Create nodes (index 0 reserved for ROOT)
    nodes = [TNode(
        index=0,
        word="ROOT",
        lemma="ROOT",
        pos="ROOT",
        tag="ROOT",
        dep="ROOT",
        head_index=-1
    )]

    for token in doc:
        node = TNode(
            index=token.i + 1,  # Offset by 1 (ROOT is 0)
            word=token.text,
            lemma=token.lemma_,
            pos=token.pos_,
            tag=token.tag_,
            dep=token.dep_,
            head_index=token.head.i + 1 if token.head.i >= 0 else 0
        )
        nodes.append(node)

    # Build parent-child relationships
    for node in nodes[1:]:  # Skip ROOT
        if 0 <= node.head_index < len(nodes):
            parent = nodes[node.head_index]
            parent.add_child(node)

    # Find actual root (sentence head) and attach to ROOT
    for node in nodes[1:]:
        if node.dep == "ROOT" or node.head_index == node.index:
            nodes[0].add_child(node)
            node.head_index = 0

    return nodes


def tnodes_to_tree_str(nodes: List[TNode], root_index: int = 0, indent: int = 0, visited: set = None) -> str:
    """
    Convert TNodes to tree string for debugging.

    Args:
        nodes: List of TNodes
        root_index: Index of root node
        indent: Current indentation level
        visited: Set of visited indices (for cycle detection)

    Returns:
        Tree string representation
    """
    if visited is None:
        visited = set()

    if root_index >= len(nodes) or root_index in visited:
        return ""

    visited.add(root_index)
    node = nodes[root_index]
    result = "  " * indent + f"{node.word} --{node.get_dep()}--> (head={node.head_index})\n"

    for child in node.children:
        result += tnodes_to_tree_str(nodes, child.index, indent + 1, visited)

    return result


def find_node_by_index(nodes: List[TNode], index: int) -> Optional[TNode]:
    """Find node by index."""
    if 0 <= index < len(nodes):
        return nodes[index]
    return None


def find_nodes_by_dep(nodes: List[TNode], dep: str) -> List[TNode]:
    """Find all nodes with given dependency."""
    return [n for n in nodes if n.get_dep() == dep]


def find_nodes_by_pos(nodes: List[TNode], pos: str) -> List[TNode]:
    """Find all nodes with given POS."""
    return [n for n in nodes if n.pos == pos]


def find_nodes_by_lemma(nodes: List[TNode], lemma: str) -> List[TNode]:
    """Find all nodes with given lemma."""
    return [n for n in nodes if n.lemma.lower() == lemma.lower()]
