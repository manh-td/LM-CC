import math
import re

from src.ast import is_syntactic_delimiter, build_delimiter_offsets
from src.tokenizer import tokenize_code
from src.hierarche import build_segments

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
# Compute threshold tau
# -----------------------------
def compute_tau(tokens, percentile):
    entropies = []
    for token in tokens:
        entropies.append(token.get("entropy"))

    values = sorted(entropies)
    count = len(values)
    index = math.ceil(count * percentile / 100.0) - 1
    index = max(0, min(index, count - 1))
    return values[index]


def segment_complexity(segment):
    entropies = [t["entropy"] for t in segment if t["entropy"] is not None]

    if not entropies:
        return 0.0

    return sum(entropies) / len(entropies)


def compute_lm_cc(segments, alpha):
    total = 0.0

    for depth, seg in enumerate(segments, start=1):
        c = segment_complexity(seg)
        weight = 1 + alpha * (depth - 1)
        total += weight * c

    return total


# -----------------------------
# Main LM-CC Algorithm
# -----------------------------
def lm_cc(code: str, tau=-1.0, alpha=0.5, model_name="gpt2"):
    # Step 1: preprocess
    code = remove_comments(code)

    # Step 2: entropy computation
    tokens = tokenize_code(code, model_name)

    # Compute Tau
    if tau < 0:
        tau = compute_tau(tokens, percentile=67.0)

    # Step 3: identify boundaries
    boundary_ids = []
    offsets = build_delimiter_offsets(code, "python")

    for token in tokens:
        if token.get("entropy") > tau or is_syntactic_delimiter(token, offsets):
            if token.get("token_id") not in boundary_ids:
                boundary_ids.append(token.get("token_id"))

    # Step 4: build semantic hierarchy
    segments = build_segments(tokens, boundary_ids)

    # Step 5: compute LM-CC
    lmcc = compute_lm_cc(segments, alpha)

    return lmcc


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    sample_code = """n = int(input())
if n % 2 == 0:
    print(n // 2 - 1)
else:
    print(n // 2)
    """
    score = lm_cc(sample_code, tau=5.0, alpha=0.6)
    print("LM-CC score:", round(score, 2))
    
    sample_code = """for i in range(0, b * m + 1, m):
    x=i
    y=b-i/m
    y = int(y)
    s = ((1 + y) * y // 2) * (x + 1)
    s += ((1 + x) * x // 2) * (y + 1)
    sum_s.append(s)
print(max(sum_s))
    """
    score = lm_cc(sample_code, tau=5.0, alpha=0.6)
    print("LM-CC score:", round(score, 2))