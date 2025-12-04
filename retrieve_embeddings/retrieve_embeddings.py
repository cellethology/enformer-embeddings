"""
Module for retrieving embeddings from Enformer model.

This module provides functions to initialize an Enformer model and retrieve
embeddings for DNA sequences.
python retrieve_embeddings/retrieve_embeddings.py \
    --input-file test_files/test.fasta \
    --output-file test_files/embeddings.npz

python -m retrieve_embeddings.cli \
    --input-file test_files/test.fasta \
    --output-file test_files/embeddings.npz
"""

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from enformer_pytorch import Enformer, seq_indices_to_one_hot
from retrieve_embeddings.util import (
    fasta_sequences_to_tensors,
    read_fasta_sequences,
    dna_string_to_indices,
)


def create_enformer_model(
    dim: int = 1536,
    depth: int = 11,
    heads: int = 8,
    output_heads: Optional[Dict[str, int]] = None,
    target_length: int = 896,
    use_tf_gamma: bool = False,
) -> Enformer:
    """
    Create an Enformer model with specified hyperparameters.

    Args:
        dim: Dimension of the model. Defaults to 1536.
        depth: Number of transformer layers. Defaults to 11.
        heads: Number of attention heads. Defaults to 8.
        output_heads: Dictionary mapping species names to number of output heads.
                     Defaults to {'human': 5313, 'mouse': 1643}.
        target_length: Target sequence length for output. Defaults to 896.
        use_tf_gamma: Whether to use TensorFlow gamma parameter. Defaults to False.

    Returns:
        Enformer: Initialized Enformer model in evaluation mode.

    Example:
        >>> model = create_enformer_model()
        >>> print(type(model))
        <class 'enformer_pytorch.enformer.Enformer'>
    """
    if output_heads is None:
        output_heads = dict(human=5313, mouse=1643)

    model = Enformer.from_hparams(
        dim=dim,
        depth=depth,
        heads=heads,
        output_heads=output_heads,
        target_length=target_length,
        use_tf_gamma=use_tf_gamma,
    )
    model.eval()
    return model


def retrieve_embeddings(
    sequence_indices: torch.Tensor,
    model: Optional[Enformer] = None,
    batch_size: int = 8,
    mean_pool: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Retrieve embeddings from Enformer model for given sequence indices.

    Args:
        sequence_indices: Tensor of shape (batch_size, sequence_length) containing
                         integer indices (0-4 for A, C, G, T, N).
                         Expected sequence length is 196,608.
        model: Enformer model instance. If None, creates a new model with default parameters.

    Returns:
        Tuple containing:
            - embeddings: Tensor of shape (batch_size, target_length, embedding_dim).
                         Default shape is (batch_size, 896, 3072).
            - outputs: Optional tensor of model outputs if return_outputs=True, else None.

    Raises:
        ValueError: If sequence_indices has incorrect shape or dimensions.

    Example:
        >>> model = create_enformer_model()
        >>> seq = torch.randint(0, 5, (1, 196_608))
        >>> embeddings, _ = retrieve_embeddings(seq, model)
        >>> print(embeddings.shape)
        torch.Size([1, 896, 3072])
    """
    if sequence_indices.dim() != 2:
        raise ValueError(
            f"Expected 2D tensor (batch, seq_len), got {sequence_indices.dim()}D"
        )

    if model is None:
        model = create_enformer_model()
    
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(sequence_indices), batch_size),
            desc="Extracting Features",
            unit="batch",
        ):
            batch_sequences = sequence_indices[i : i + batch_size]
            if not batch_sequences:
                continue
            while True:
                try:
                    _, embeddings = model(batch_sequences, return_embeddings=True)
                    sequence_embeddings = embeddings.mean(dim=-2)
                    all_embeddings.append(sequence_embeddings.cpu().numpy())
                    break
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"CUDA Out of Memory on batch starting at index {i}. Retrying batch."
                    )
                    torch.cuda.empty_cache()
                
    if not all_embeddings:
        return np.array([])

    return np.concatenate(all_embeddings, axis=0)

def retrieve_embeddings_from_fasta(
    fasta_path: str | Path,
    model: Optional[Enformer] = None,
    center_sequences: bool = True,
    window_size: int = 196_608,
    pad_value: str | int = "N",
    return_outputs: bool = False,
    return_sequence_ids: bool = True,
    save_path: Optional[str | Path] = None,
    mean_pool: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]:
    """
    Read sequences from FASTA file and retrieve embeddings.

    This is a high-level convenience function that combines FASTA reading
    with embedding retrieval.

    Args:
        fasta_path: Path to the FASTA file.
        model: Enformer model instance. If None, creates a new model with default parameters.
        center_sequences: If True, center sequences in window_size window with padding.
                         Defaults to True.
        window_size: Target window size for centering. Defaults to 196,608.
        pad_value: Value to use for padding. Can be 'N', '-', or -1 (integer).
                   If 'N' or '-', pads with character then converts to indices.
                   If -1, pads tensor directly with -1. Defaults to 'N'.
        return_outputs: If True, also returns model outputs. Defaults to False.
        return_sequence_ids: If True, returns list of sequence IDs from FASTA file.
                            Defaults to True.
        save_path: Optional path to save embeddings in compressed npz format.
                  If provided, saves embeddings and IDs as np.savez_compressed.
                  Defaults to None.
        mean_pool: If True, apply mean pooling across the embedding dimension.
                  Reduces shape from (num_sequences, 896, 3072) to (num_sequences, 896).
                  Defaults to False.

    Returns:
        Tuple containing:
            - embeddings: Tensor of shape (num_sequences, target_length, embedding_dim)
                         if mean_pool=False, or (num_sequences, target_length) if mean_pool=True.
                         Default shape is (num_sequences, 896, 3072) or (num_sequences, 896).
            - outputs: Optional tensor of model outputs if return_outputs=True, else None.
            - sequence_ids: Optional list of sequence IDs if return_sequence_ids=True, else None.

    Raises:
        FileNotFoundError: If the FASTA file does not exist.
        ValueError: If the FASTA file is empty or contains no valid sequences.

    Example:
        >>> embeddings, _, ids = retrieve_embeddings_from_fasta("test.fasta")
        >>> print(embeddings.shape)
        torch.Size([3, 896, 3072])
        >>> print(ids)
        ['seq1', 'seq2', 'seq3']
        >>> embeddings, _, ids = retrieve_embeddings_from_fasta(
        ...     "test.fasta", pad_value=-1, save_path="embeddings.npz", mean_pool=True
        ... )
        >>> print(embeddings.shape)
        torch.Size([3, 896])
    """
    # Read sequences from FASTA and convert to tensors
    sequence_ids, sequence_tensors = fasta_sequences_to_tensors(
        fasta_path,
        center_sequences=center_sequences,
        window_size=window_size,
        pad_value=pad_value,
    )

    # Retrieve embeddings
    embeddings = retrieve_embeddings(
        sequence_tensors, model=model, batch_size=8, mean_pool=mean_pool
    )   

    # Save to npz if save_path is provided
    if save_path is not None:
        # Convert sequence_ids to numpy array
        ids_np = np.array(sequence_ids)
        # Save in compressed npz format
        np.savez_compressed(save_path, ids=ids_np, embeddings=embeddings)

    if return_sequence_ids:
        return embeddings, sequence_ids
    else:
        return embeddings, None


if __name__ == "__main__":
    # When run as script, delegate to CLI module
    from retrieve_embeddings.cli import main

    main()
