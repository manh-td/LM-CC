# LM-CC: Language Model Code Complexity

I'm not part of the team, but since I can't find their replication package, this is my attempt to create a script to calculate LM-CC from the paper Rethinking Code Complexity Through the Lens of Large Language Models. Please feel free to make a PR to help improve this code, or an Issue if you find any.

## Project Structure

- **`src/`** - Core modules
  - `main.py` - Entry point; Main Pipeline
  - `tokenizer.py` - Tokenize code; Calculate entrophy
  - `delimiter.py` - Delimiter Offset
  - `config.py` - Configuration
  - `hierarche.py` - Build Hierarchy Tree

- **`scripts/`** - Test and utility scripts
  - `main.sh` - Main execution script
  - `test.tokenizer.sh` - Tokenizer test
  - `test.ast.sh` - AST test

- **`example-code/`** - Example code for testing
  - `simple.py` - Simple Python example
  - `complex.py` - Complex Python examples

## Installation

1. Clone the repository
2. Run: `docker compose up -d`
3. Enter container: `docker exec -it lm-cc /bin/bash`

## Testing

Run test scripts from `scripts/`:

```bash
bash scripts/test.tokenizer.sh
bash scripts/test.ast.sh
bash scripts/test.checker.sh
```

## Usage

```bash
python src/main.py
```

Refer to `scripts/` for usage examples.

## The Paper
[Rethinking Code Complexity Through the Lens of Large Language Models](https://arxiv.org/abs/2602.07882)

Please cite them if you use this code
```
@article{xie2026rethinking,
  title={Rethinking Code Complexity Through the Lens of Large Language Models},
  author={Xie, Chen and Shi, Yuling and Gu, Xiaodong and Shen, Beijun},
  journal={arXiv preprint arXiv:2602.07882},
  year={2026}
}
```