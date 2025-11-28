# Enformer Embeddings

Enformer Embeddings is a Python framework for extracting embeddings from DNA sequences using the Enformer model. This framework allows you to process FASTA files and extract high-dimensional embeddings that can be used for downstream analysis tasks.

## Overview

The Enformer Embeddings framework provides tools to:
- Extract embeddings from FASTA sequences using the Enformer model
- Process sequences with automatic centering and padding
- Apply mean pooling for dimensionality reduction
- Save embeddings in compressed NumPy format for easy loading

## Prerequisites

- Python 3.11 or higher
- `uv` package manager (install from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv))
- CUDA-capable GPU (optional, but recommended for faster inference)

## Environment Setup

### 1. Install uv (if not already installed)

```bash
# Or using pip
pip install uv
```

### 2. Create Virtual Environment and Install Dependencies

Run the environment setup command:

```bash
uv sync
source .venv/bin/activate
```

This will install all required dependencies including:
- `enformer-pytorch`: Enformer model implementation
- `biopython`: FASTA file parsing
- `pandas`: Data processing utilities
- `numpy`: Numerical operations

## Usage

### Extract Embeddings from FASTA Sequences

The main script for extracting embeddings can be run like below:

#### Module Execution

```bash
python -m retrieve_embeddings.cli \
    --input-file test_files/test.fasta \
    --output-file output/embeddings.npz
```

#### Full Command with All Options

```bash
python -m retrieve_embeddings.cli \
    --input-file <path-to-input.fasta> \
    --output-file <path-to-output.npz> \
    --window-size 196608 \
    --pad-value N \
    --mean-pool \
    --no-center
```

#### Command-Line Arguments

- `--input-file` (required): Path to input FASTA file containing DNA sequences
- `--output-file` (required): Path to output `.npz` file where embeddings will be saved
- `--window-size` (optional): Window size for sequence centering. Defaults to 196608 (Enformer requirement)
- `--pad-value` (optional): Padding value when sequences are shorter than window size. Options:
  - `N`: Pad with 'N' characters (default)
  - `-`: Pad with gap character '-'
  - `-1`: Pad with -1 index value
- `--mean-pool` (optional): Apply mean pooling across the embedding dimension. Reduces shape from `(N, 896, 3072)` to `(N, 896)`
- `--no-center` (optional): Disable sequence centering. Sequences must be exactly `window_size` in length

#### Output Format

The script outputs a compressed NumPy archive (`.npz`) file containing:
- `ids`: Array of sequence IDs from the FASTA file
- `embeddings`: Array of embeddings with shape:
  - `(num_sequences, 896, 3072)` without mean pooling
  - `(num_sequences, 896)` with mean pooling (`--mean-pool` flag)

#### Examples

```bash
# Basic usage with default settings
python -m retrieve_embeddings.cli \
    --input-file test_files/test.fasta \
    --output-file output/embeddings.npz

# With mean pooling for reduced dimensionality
python -m retrieve_embeddings.cli \
    --input-file test_files/test.fasta \
    --output-file output/embeddings.npz \
    --mean-pool

# Using gap character padding
python -m retrieve_embeddings.cli \
    --input-file test_files/test.fasta \
    --output-file output/embeddings.npz \
    --pad-value -

# Disable centering (sequences must be exactly 196608 bp)
python -m retrieve_embeddings.cli \
    --input-file test_files/test.fasta \
    --output-file output/embeddings.npz \
    --no-center
```

### Loading Embeddings

You can load the saved embeddings in Python:

```python
import numpy as np

# Load embeddings
data = np.load('output/embeddings.npz')
sequence_ids = data['ids']
embeddings = data['embeddings']

print(f"Loaded {len(sequence_ids)} sequences")
print(f"Embeddings shape: {embeddings.shape}")  # (num_sequences, 896, 3072) or (num_sequences, 896)
print(f"Sequence IDs: {sequence_ids}")
```

### Programmatic Usage

You can also use the framework programmatically in Python:

```python
from retrieve_embeddings import retrieve_embeddings_from_fasta, create_enformer_model

# Create model
model = create_enformer_model()

# Retrieve embeddings
embeddings, _, sequence_ids = retrieve_embeddings_from_fasta(
    fasta_path="test_files/test.fasta",
    model=model,
    center_sequences=True,
    window_size=196608,
    pad_value="N",
    mean_pool=False,
    save_path="output/embeddings.npz"
)

print(f"Processed {len(sequence_ids)} sequences")
print(f"Embeddings shape: {embeddings.shape}")
```

## Project Structure

```
enformer-embeddings/
├── retrieve_embeddings/        # Main package
│   ├── __init__.py              # Package initialization
│   ├── retrieve_embeddings.py   # Core embedding functions
│   ├── cli.py                   # Command-line interface
│   └── util.py                  # Utility functions (FASTA parsing, validation)
├── test/                        # Unit tests
│   ├── test_pre_embeddings.py   # Tests for preprocessing functions
│   └── test_get_embeddings.py   # Tests for embedding extraction
├── test_files/                  # Example files
│   ├── test.fasta               # Example FASTA input
│   └── embeddings.npz           # Example output
├── pyproject.toml               # Project dependencies and configuration
└── README.md                    # This file
```

## Sequence Processing

### Sequence Validation

The framework automatically validates sequences:
- **Length check**: Sequences longer than `window_size` will raise a `ValueError`
- **Character validation**: Only valid nucleotide characters (A, C, G, T, N, -) are allowed

### Sequence Centering

By default, sequences are centered in a window of size 196608:
- **Shorter sequences**: Padded on both sides with the specified padding value
- **Exact length**: No padding needed
- **Longer sequences**: Raises an error (use `--no-center` if sequences are pre-processed)

### Padding Options

- **`N` (default)**: Pads with 'N' characters, converted to index 4
- **`-`**: Pads with gap character '-', converted to index -1
- **`-1`**: Directly pads tensor with -1 index value

## Testing

Run the test suite to verify your installation:

```bash
pytest test/
```

Or run specific test files:

```bash
pytest test/test_pre_embeddings.py -v
```

## Troubleshooting

### Sequence Length Errors

If you encounter `ValueError: Sequence length exceeds window size`, ensure that:
1. Sequences are at most 196608 base pairs long (or your specified `window_size`)
2. Or use `--no-center` flag if sequences are already pre-processed to exact length

### Invalid Character Errors

If you encounter `ValueError: Invalid character`, ensure that:
1. FASTA sequences contain only valid nucleotides: A, C, G, T, N, or -
2. Sequences are properly formatted (no special characters)

### Import Errors

If you encounter import errors, make sure:
1. The package is installed: `uv pip install -e .`
2. You're in the correct virtual environment
3. All dependencies are installed: `uv sync`

### GPU/CPU Usage

The framework will automatically use GPU if available (CUDA), otherwise it will fall back to CPU. Processing is slower on CPU but will work correctly.

## Dependencies

- `enformer-pytorch>=0.8.11`: Enformer model implementation
- `biopython>=1.83`: FASTA file parsing
- `pandas>=2.3.1`: Data processing
- `numpy`: Numerical operations (included with PyTorch)
- `torch`: Deep learning framework (included with enformer-pytorch)
