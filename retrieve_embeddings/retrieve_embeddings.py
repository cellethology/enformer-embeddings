"""
Module for retrieving embeddings from Enformer model.

This module provides functions to initialize an Enformer model and retrieve
embeddings for DNA sequences.

Example usage:
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
from typing import Optional, Tuple, Dict, List, Union
from enformer_pytorch import from_pretrained, seq_indices_to_one_hot
from retrieve_embeddings.util import (
    fasta_sequences_to_tensors,
    read_fasta_sequences,
    dna_string_to_indices,
)


def _choose_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Internal helper to pick a device.

    If device is provided (str or torch.device) it is returned as torch.device.
    If None, prefers CUDA when available.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def create_enformer_model(
    dim: int = 1536,
    depth: int = 11,
    heads: int = 8,
    output_heads: Optional[Dict[str, int]] = None,
    target_length: int = 896,
    use_tf_gamma: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> Enformer:
    """
    Create an Enformer model with specified hyperparameters and move it to the chosen device.

    Args:
        dim: Dimension of the model. Defaults to 1536.
        depth: Number of transformer layers. Defaults to 11.
        heads: Number of attention heads. Defaults to 8.
        output_heads: Dictionary mapping species names to number of output heads.
                     Defaults to {'human': 5313, 'mouse': 1643}.
        target_length: Target sequence length for output. Defaults to 896.
        use_tf_gamma: Whether to use TensorFlow gamma parameter. Defaults to False.
        device: Optional device to move model to ('cuda', 'cpu', or torch.device). If None,
                will use CUDA when available.

    Returns:
        Enformer: Initialized Enformer model in evaluation mode on the requested device.
    """
    if output_heads is None:
        output_heads = dict(human=5313, mouse=1643)

    # model = Enformer.from_hparams(
    #     dim=dim,
    #     depth=depth,
    #     heads=heads,
    #     output_heads=output_heads,
    #     target_length=target_length,
    #     use_tf_gamma=use_tf_gamma,
    # )
    model = from_pretrained("EleutherAI/enformer-official-rough")
    dev = _choose_device(device)
    model.to(dev)
    model.eval()
    
    return model


def retrieve_embeddings(
    sequence_indices: torch.Tensor,
    model: Optional[Enformer] = None,
    batch_size: int = 8,
    mean_pool: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    """
    Retrieve embeddings from Enformer model for given sequence indices.

    This function will move the model and input batches to GPU if CUDA is available
    (or to the provided device).

    Args:
        sequence_indices: Tensor of shape (num_sequences, sequence_length) containing
                         integer indices (0-4 for A, C, G, T, N).
        model: Enformer model instance. If None, creates a new model with default parameters.
        batch_size: Number of sequences per batch.
        mean_pool: If True, apply mean pooling across the second-to-last dimension as the
                   original code did.
        device: Optional device to use ('cuda' or 'cpu' or torch.device). If None, uses CUDA
                when available.

    Returns:
        NumPy array of concatenated embeddings with shape (num_sequences, target_length, embedding_dim)
        or (num_sequences, target_length) if mean_pool=True and pooling reduces the last dimension.
    """
    if sequence_indices.dim() != 2:
        raise ValueError(
            f"Expected 2D tensor (batch, seq_len), got {sequence_indices.dim()}D"
        )

    dev = _choose_device(device)

    if model is None:
        model = create_enformer_model(device=dev)
    else:
        # ensure the model is on the chosen device
        try:
            model.to(dev)
        except Exception:
            # Some models may not implement .to gracefully; ignore if it fails
            pass

    all_embeddings = []
    num_sequences = len(sequence_indices)

    with torch.no_grad():
        for start in tqdm(
            range(0, num_sequences, batch_size),
            desc="Extracting Features",
            unit="batch",
        ):
            batch_sequences = sequence_indices[start : start + batch_size]

            # ensure dtype is appropriate for model input (long indices)
            if not batch_sequences.dtype.is_floating_point:
                batch_sequences = batch_sequences.long()

            # Move batch to device
            batch_sequences = batch_sequences.to(dev)

            retried_on_cpu = False
            while True:
                try:
                    # The model is expected to return (outputs, embeddings) when return_embeddings=True.
                    # We call with the batch on the same device as the model.
                    outputs, embeddings = model(batch_sequences, return_embeddings=True)
                    # original code averaged over dim -2; keep that behavior
                    if mean_pool:
                        sequence_embeddings = embeddings.mean(dim=-2)
                    else:
                        sequence_embeddings = embeddings
                    all_embeddings.append(sequence_embeddings.cpu().numpy())
                    break
                except RuntimeError as e:
                    # Catch CUDA OOM and fallback to CPU for this batch (move model too).
                    # Use string check because different torch versions raise different types/messages.
                    msg = str(e).lower()
                    if "out of memory" in msg and torch.cuda.is_available() and not retried_on_cpu:
                        print(
                            f"CUDA Out of Memory on batch starting at index {start}. "
                            "Falling back to CPU for this batch and clearing CUDA cache."
                        )
                        torch.cuda.empty_cache()
                        # move model to cpu and run this batch on cpu
                        try:
                            model.to("cpu")
                        except Exception:
                            pass
                        batch_sequences = batch_sequences.cpu()
                        dev = torch.device("cpu")
                        retried_on_cpu = True
                        continue
                    # If it's not OOM or already retried on CPU, re-raise
                    raise

    if not all_embeddings:
        # return empty numpy array of shape (0,)
        return np.array([])

    return np.concatenate(all_embeddings, axis=0)


def retrieve_embeddings_from_fasta(
    fasta_path: Union[str, Path],
    model: Optional[Enformer] = None,
    center_sequences: bool = True,
    window_size: int = 196_608,
    pad_value: Union[str, int] = "N",
    return_outputs: bool = False,
    return_sequence_ids: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    mean_pool: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
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
        return_outputs: Reserved for compatibility (not currently returned separately).
        return_sequence_ids: If True, returns list of sequence IDs from FASTA file.
                            Defaults to True.
        save_path: Optional path to save embeddings in compressed npz format.
                  If provided, saves embeddings and IDs as np.savez_compressed.
                  Defaults to None.
        mean_pool: If True, apply mean pooling across the embedding dimension.
                  Defaults to False.
        device: Optional device to use ('cuda' or 'cpu' or torch.device). If None, uses CUDA when available.

    Returns:
        Tuple containing:
            - embeddings: NumPy array of embeddings (num_sequences, target_length, embedding_dim)
                         or (num_sequences, target_length) if mean_pool=True.
            - sequence_ids: Optional list of sequence IDs if return_sequence_ids=True, else None.
    """
    # Read sequences from FASTA and convert to tensors
    sequence_ids, sequence_tensors = fasta_sequences_to_tensors(
        fasta_path,
        center_sequences=center_sequences,
        window_size=window_size,
        pad_value=pad_value,
    )

    # Retrieve embeddings (this will move model and batches to GPU when available)
    embeddings = retrieve_embeddings(
        sequence_tensors, model=model, batch_size=8, mean_pool=mean_pool, device=device
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
