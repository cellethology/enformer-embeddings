"""
Utility functions for sequence processing and FASTA file handling.

This module provides functions to read FASTA files and convert DNA sequences
to the format required by Enformer models.
"""

import torch
from pathlib import Path
from typing import List, Tuple, Dict
from Bio import SeqIO


def validate_sequence(
    sequence: str,
    window_size: int | None = None,
    allowed_chars: str | None = None,
) -> None:
    """
    Validate a DNA sequence string.

    Checks that:
    1. Sequence length does not exceed window_size (if provided)
    2. Sequence contains only valid nucleotide characters

    Args:
        sequence: DNA sequence string to validate.
        window_size: Maximum allowed sequence length. If None, no length check is performed.
        allowed_chars: String of allowed characters. Defaults to 'ACGTN-'.

    Raises:
        ValueError: If sequence length exceeds window_size or contains invalid characters.

    Example:
        >>> validate_sequence("ACGT", window_size=10)
        >>> validate_sequence("ACGTX", window_size=10)  # Raises ValueError
        Traceback (most recent call last):
            ...
        ValueError: Invalid character 'X' found in sequence. Allowed characters: ACGTN-
    """
    if allowed_chars is None:
        allowed_chars = "ACGTN-"

    # Check for invalid characters
    sequence_upper = sequence.upper()
    invalid_chars = set(sequence_upper) - set(allowed_chars.upper())
    if invalid_chars:
        invalid_chars_str = ", ".join(sorted(f"'{c}'" for c in invalid_chars))
        raise ValueError(
            f"Invalid character(s) {invalid_chars_str} found in sequence. "
            f"Allowed characters: {allowed_chars}"
        )

    # Check sequence length
    if window_size is not None and len(sequence) > window_size:
        raise ValueError(
            f"Sequence length ({len(sequence)}) exceeds window size ({window_size}). "
            f"Sequence must be at most {window_size} characters long."
        )


def read_fasta_sequences(
    fasta_path: str | Path,
) -> List[Tuple[str, str]]:
    """
    Read sequences from a FASTA file.

    Args:
        fasta_path: Path to the FASTA file.

    Returns:
        List of tuples containing (sequence_id, sequence_string).

    Raises:
        FileNotFoundError: If the FASTA file does not exist.
        ValueError: If the FASTA file is empty or contains no valid sequences.

    Example:
        >>> sequences = read_fasta_sequences("test.fasta")
        >>> print(f"Found {len(sequences)} sequences")
        Found 3 sequences
        >>> print(sequences[0][0])  # First sequence ID
        seq1
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    sequences = []
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_str = str(record.seq).upper()
            sequences.append((record.id, seq_str))
    except Exception as e:
        raise ValueError(f"Error parsing FASTA file: {e}") from e

    if len(sequences) == 0:
        raise ValueError(f"No sequences found in FASTA file: {fasta_path}")

    return sequences


def dna_string_to_indices(
    sequence: str,
    validate: bool = True,
    window_size: int | None = None,
) -> torch.Tensor:
    """
    Convert DNA sequence string to tensor of indices.

    Mapping: A=0, C=1, G=2, T=3, N=4, -=-1 (gap character)

    Args:
        sequence: DNA sequence string (case-insensitive).
        validate: If True, validates the sequence before conversion. Defaults to True.
        window_size: Maximum allowed sequence length for validation. Defaults to None.

    Returns:
        torch.Tensor: Tensor of shape (sequence_length,) with integer indices.

    Raises:
        ValueError: If validate=True and sequence contains invalid characters or exceeds window_size.

    Example:
        >>> seq = "ACGTN-"
        >>> indices = dna_string_to_indices(seq)
        >>> print(indices)
        tensor([0, 1, 2, 3, 4, -1])
    """
    if validate:
        validate_sequence(sequence, window_size=window_size)

    mapping: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "-": -1}

    indices = [mapping.get(base.upper(), 4) for base in sequence]
    return torch.tensor(indices, dtype=torch.long)

def center_sequence_tensor_in_window(
    sequence_tensor: torch.Tensor,
    window_size: int = 196_608,
    pad_value: int = -1,
) -> torch.Tensor:
    """
    Center a sequence tensor in a window of specified size, padding with specified value.

    If the sequence is longer than the window, a ValueError is raised.
    If shorter, it will be padded with the specified padding value on both sides.

    Args:
        sequence_tensor: Input sequence tensor of shape (sequence_length,).
        window_size: Target window size. Defaults to 196,608 (Enformer requirement).
        pad_value: Integer value to use for padding. Defaults to -1.

    Returns:
        torch.Tensor: Centered sequence tensor of exactly window_size length.

    Raises:
        ValueError: If sequence length exceeds window_size.

    Example:
        >>> seq = torch.tensor([0, 1, 2, 3])  # ACGT
        >>> centered = center_sequence_tensor_in_window(seq, window_size=10, pad_value=-1)
        >>> print(centered)
        tensor([-1, -1, -1,  0,  1,  2,  3, -1, -1, -1])
    """
    seq_len = sequence_tensor.shape[0]

    if seq_len > window_size:
        raise ValueError(
            f"Sequence length ({seq_len}) exceeds window size ({window_size}). "
            f"Sequence must be at most {window_size} elements long."
        )

    if seq_len == window_size:
        # No padding needed
        return sequence_tensor

    # Calculate padding needed
    pad_total = window_size - seq_len
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    # Create padding tensors
    left_pad = torch.full((pad_left,), pad_value, dtype=sequence_tensor.dtype)
    right_pad = torch.full((pad_right,), pad_value, dtype=sequence_tensor.dtype)

    # Concatenate: left_pad + sequence + right_pad
    centered_tensor = torch.cat([left_pad, sequence_tensor, right_pad])

    return centered_tensor


def fasta_sequences_to_tensors(
    fasta_path: str | Path,
    center_sequences: bool = True,
    window_size: int = 196_608,
    pad_value: str | int = "N",
) -> Tuple[List[str], torch.Tensor]:
    """
    Read FASTA file and convert sequences to tensor format.

    Args:
        fasta_path: Path to the FASTA file.
        center_sequences: If True, center sequences in window_size window with padding.
                         Defaults to True.
        window_size: Target window size for centering. Defaults to 196,608.
        pad_value: Value to use for padding. Can be 'N', '-', or -1 (integer).
                   If 'N', pads tensor with 4 (N index).
                   If '-', pads tensor with -1.
                   If -1, pads tensor directly with -1. Defaults to 'N'.

    Returns:
        Tuple containing:
            - sequence_ids: List of sequence IDs from FASTA file.
            - sequence_tensors: Tensor of shape (num_sequences, sequence_length) with indices.

    Example:
        >>> ids, tensors = fasta_sequences_to_tensors("test.fasta")
        >>> print(tensors.shape)
        torch.Size([3, 196608])
        >>> ids, tensors = fasta_sequences_to_tensors("test.fasta", pad_value="-")
        >>> print(tensors.shape)
        torch.Size([3, 196608])
        >>> ids, tensors = fasta_sequences_to_tensors("test.fasta", pad_value=-1)
        >>> print(tensors.shape)
        torch.Size([3, 196608])
    """
    sequences = read_fasta_sequences(fasta_path)
    sequence_ids = []
    tensor_list = []

    # Determine the integer pad value to use
    if isinstance(pad_value, int):
        tensor_pad_value = pad_value
    elif pad_value == "N":
        tensor_pad_value = 4  # N maps to index 4
    elif pad_value == "-":
        tensor_pad_value = -1  # - maps to -1
    else:
        raise ValueError(
            f"pad_value must be 'N', '-', or an integer (typically -1), got '{pad_value}'"
        )

    for seq_id, seq_str in sequences:
        sequence_ids.append(seq_id)

        # Convert sequence to tensor indices (validates sequence and window size)
        indices = dna_string_to_indices(seq_str, validate=True, window_size=window_size)

        if center_sequences:
            # Center the tensor using center_sequence_tensor_in_window
            indices = center_sequence_tensor_in_window(
                indices, window_size=window_size, pad_value=tensor_pad_value
            )

        tensor_list.append(indices)

    sequence_tensors = torch.stack(tensor_list)

    return sequence_ids, sequence_tensors

