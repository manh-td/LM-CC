import re
from collections import deque
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Utility: Remove comments
# -----------------------------
def remove_comments(code: str) -> str:
    # Remove single-line comments
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"//.*", "", code)
    # Remove multi-line comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return code


# -----------------------------
# Utility: Simple tokenization
# -----------------------------
def simple_tokenize(code: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", code, re.UNICODE)


# -----------------------------
# Compute token-level entropy
# -----------------------------
def compute_token_entropy(code: str, model, tokenizer):
    inputs = tokenizer(code, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = F.softmax(logits, dim=-1)

    entropies = []
    for i in range(1, logits.shape[1]):
        dist = probs[0, i - 1]
        entropy = -(dist * torch.log(dist + 1e-12)).sum().item()
        entropies.append(entropy)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:]
    return tokens, entropies


# -----------------------------
# Detect syntactic boundaries
# -----------------------------
def is_syntactic_boundary(token):
    return token in {"def", "class", "{", "}", "for", "while", "if"}


# -----------------------------
# Semantic Unit Node
# -----------------------------
@dataclass
class Unit:
    start: int
    end: int
    children: List["Unit"] = field(default_factory=list)

    def depth(self):
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def branching_factor(self):
        return len(self.children)


# -----------------------------
# Partition by boundaries
# -----------------------------
def partition_units(tokens, boundaries):
    units = []
    start = 0
    for b in sorted(boundaries):
        if b > start:
            units.append((start, b))
            start = b
    if start < len(tokens):
        units.append((start, len(tokens)))
    return units


# -----------------------------
# Build hierarchy (BFS)
# -----------------------------
def build_hierarchy(tokens, boundaries):
    root = Unit(0, len(tokens))
    queue = deque([(root, 0, len(tokens))])

    while queue:
        parent, s, e = queue.popleft()
        sub_bounds = [b for b in boundaries if s < b < e]

        if not sub_bounds:
            continue

        ranges = partition_units(tokens[s:e], [b - s for b in sub_bounds])
        for rs, re_ in ranges:
            child = Unit(s + rs, s + re_)
            parent.children.append(child)
            queue.append((child, s + rs, s + re_))

    return root


# -----------------------------
# Compute LM-CC score
# -----------------------------
def compute_lmcc(root: Unit, alpha=0.5):
    total_score = 0
    queue = deque([root])

    while queue:
        unit = queue.popleft()
        b = unit.branching_factor()
        d = unit.depth()
        total_score += alpha * b + (1 - alpha) * d
        queue.extend(unit.children)

    return total_score


# -----------------------------
# Main LM-CC Algorithm
# -----------------------------
def lm_cc(code: str, tau=5.0, alpha=0.5, model_name="gpt2"):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Step 1: preprocess
    code = remove_comments(code)

    # Step 2: entropy computation
    tokens, entropies = compute_token_entropy(code, model, tokenizer)

    # Step 3: identify boundaries
    boundaries = set()
    for i, (tok, H) in enumerate(zip(tokens, entropies)):
        if H > tau or is_syntactic_boundary(tok):
            boundaries.add(i)

    # Step 4: build semantic hierarchy
    root = build_hierarchy(tokens, boundaries)

    # Step 5: compute LM-CC
    score = compute_lmcc(root, alpha)

    return score


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    sample_code = """
    def foo(x):
        if x > 0:
            return x
        else:
            return -x
    """

    score = lm_cc(sample_code, tau=5.0, alpha=0.6)
    print("LM-CC score:", score)