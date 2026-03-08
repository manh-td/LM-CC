from unittest.mock import patch, MagicMock
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import main as lmcc


# -----------------------------
# remove_comments
# -----------------------------
def test_remove_comments():
    code = """
    # comment
    def foo():  # inline comment
        pass
    """
    cleaned = lmcc.remove_comments(code)
    assert "#" not in cleaned
    assert "comment" not in cleaned
    assert "def foo" in cleaned
    

# -----------------------------
# compute_tau
# -----------------------------
def test_compute_tau_percentile():
    entropies = [1.0, 2.0, 3.0, 4.0, 5.0]

    tau = lmcc.compute_tau(entropies, percentile=60)

    # ceil(5 * 0.6) - 1 = ceil(3) - 1 = 2
    # sorted values -> index 2 = 3.0
    assert tau == 3.0


# -----------------------------
# syntactic boundary detection
# -----------------------------
def test_is_syntactic_boundary():
    assert lmcc.is_syntactic_boundary("def")
    assert lmcc.is_syntactic_boundary("{")
    assert not lmcc.is_syntactic_boundary("variable")


# -----------------------------
# partition_units
# -----------------------------
def test_partition_units():
    tokens = ["a", "b", "c", "d", "e"]
    boundaries = {2, 4}
    units = lmcc.partition_units(tokens, boundaries)
    assert units == [(0, 2), (2, 4), (4, 5)]


# -----------------------------
# hierarchy construction
# -----------------------------
def test_build_hierarchy():
    tokens = ["a", "b", "c", "d"]
    boundaries = {2}
    root = lmcc.build_hierarchy(tokens, boundaries)

    assert root.start == 0
    assert root.end == 4
    assert len(root.children) == 2
    assert root.children[0].start == 0
    assert root.children[0].end == 2
    assert root.children[1].start == 2
    assert root.children[1].end == 4


# -----------------------------
# compute_lmcc scoring
# -----------------------------
def test_compute_lmcc_simple_tree():
    root = lmcc.Unit(0, 4)
    child1 = lmcc.Unit(0, 2)
    child2 = lmcc.Unit(2, 4)

    root.children = [child1, child2]

    score = lmcc.compute_lmcc(root, alpha=0.5)

    # root: branching=2, depth=2
    # child1: branching=0, depth=1
    # child2: branching=0, depth=1
    #
    # total = 0.5*(2)+(0.5*2)
    #       + 0.5*(0)+(0.5*1)
    #       + 0.5*(0)+(0.5*1)
    expected = (0.5*2 + 0.5*2) + (0.5*0 + 0.5*1) + (0.5*0 + 0.5*1)

    assert score == expected


# -----------------------------
# Mocked LM test (no real model)
# -----------------------------
@patch("main.AutoTokenizer.from_pretrained")
@patch("main.AutoModelForCausalLM.from_pretrained")
def test_lm_cc_mocked_model(mock_model_class, mock_tokenizer_class):

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4]])
    }
    mock_tokenizer.convert_ids_to_tokens.return_value = ["a", "b", "c"]
    mock_tokenizer_class.return_value = mock_tokenizer

    # Mock model
    mock_model = MagicMock()
    mock_outputs = MagicMock()

    # Fake logits tensor
    mock_outputs.logits = torch.randn(1, 4, 10)
    mock_model.return_value = mock_outputs
    mock_model.eval = MagicMock()

    mock_model_class.return_value = mock_model

    code = "def foo(): pass"
    score = lmcc.lm_cc(code, tau=0.0, alpha=0.5)

    assert isinstance(score, float)
    assert score >= 0