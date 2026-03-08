import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def tokenize_code(code: str, model_name="gpt2", indent_size=4):
    """
    Tokenizes code using a Hugging Face tokenizer, maps tokens to line/column,
    computes entropy, and adds indent_level.

    Returns list of dicts:
    token, token_id, line, start_index, end_index, entropy, indent_level
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(code, return_offsets_mapping=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    offset_mapping = inputs["offset_mapping"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)

    # Split lines
    lines = code.splitlines(keepends=True)

    # Build line start offsets
    line_start_offsets = []
    offset = 0
    for l in lines:
        line_start_offsets.append(offset)
        offset += len(l)

    # Compute entropy
    entropies = []
    for i in range(1, input_ids.shape[1]):
        dist = probs[0, i - 1]
        entropy = -(dist * torch.log(dist + 1e-12)).sum().item()
        entropies.append(entropy)

    tokens_info = []

    for i, (token_id, (start_char, end_char)) in enumerate(zip(input_ids[0], offset_mapping[0])):

        if start_char == end_char:
            continue

        # Determine line
        line_num = 0
        for j, line_offset in enumerate(line_start_offsets):
            if start_char >= line_offset:
                line_num = j
            else:
                break

        start_index = start_char - line_start_offsets[line_num]
        end_index = end_char - line_start_offsets[line_num]

        token_str = tokenizer.decode([token_id])
        entropy = entropies[i - 1] if i - 1 < len(entropies) else None

        tokens_info.append({
            "token": token_str,
            "token_id": token_id.item(),
            "line": line_num + 1,
            "start_index": start_index.item(),
            "end_index": end_index.item(),
            "entropy": round(entropy, 2) if entropy is not None else None,
        })

    return tokens_info


# Example usage
code = """def add(a, b):
    return a + b
"""

if __name__ == "__main__":
    tokens_with_entropy = tokenize_code(code, model_name="gpt2")
    for t in tokens_with_entropy:
        print(t)