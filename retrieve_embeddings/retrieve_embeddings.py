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
from typing import Optional, Tuple, List, Union
from enformer_pytorch import from_pretrained
from retrieve_embeddings.util import (
    fasta_sequences_to_tensors,
    read_fasta_sequences,
)


def _choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def retrieve_embeddings(
    sequence,
    batch_size: int = 8,
    mean_pool: bool = True,
    center_sequences: bool = True,
    window_size: int = 196_608,
    pad_value: int = -1,
) -> np.ndarray:
    """
    Retrieve embeddings from Enformer model for given sequence indices.

    This function will move the model and input batches to GPU if CUDA is available.

    Returns:
        NumPy array of concatenated embeddings with shape (num_sequences, target_length, embedding_dim)
        or (num_sequences, target_length) if mean_pool=True and pooling reduces the last dimension.
    """

    model = from_pretrained("EleutherAI/enformer-official-rough")
    dev = _choose_device()
    model.to(dev)
    model.eval()
    print("Obtained pre-trained Enformer model...")

    all_seq_ids = []
    all_embeddings = []
    num_sequences = len(sequence)

    with torch.no_grad():
        for start in tqdm(
            range(0, num_sequences, batch_size),
            desc="Extracting Features",
            unit="batch",
        ):
            selected = sequence[start : start + batch_size]
            sequence_ids, sequence_tensors = fasta_sequences_to_tensors(selected, center_sequences, window_size,pad_value)

            # Move batch to device
            sequence_tensors = sequence_tensors.to(dev)

            retried_on_cpu = False
            while True:
                try:
                    _, embeddings = model(batch_sequences, return_embeddings=True)

                    if mean_pool:
                        sequence_embeddings = embeddings.mean(dim=-2)
                    else:
                        sequence_embeddings = embeddings
                    all_embeddings.append(sequence_embeddings.cpu().numpy())
                    all_seq_ids.extend(sequence_ids)
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

    return all_seq_ids, np.concatenate(all_embeddings, axis=0)


def retrieve_embeddings_from_fasta(
    fasta_path: Union[str, Path],
    center_sequences: bool = True,
    window_size: int = 196_608,
    batch_size: int = 8,
    save_path: Optional[Union[str, Path]] = None,
    mean_pool: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
    """
    Read sequences from FASTA file and retrieve embeddings.

    This is a high-level convenience function that combines FASTA reading
    with embedding retrieval.

    Args:
        fasta_path: Path to the FASTA file.
        center_sequences: If True, center sequences in window_size window with padding.
                         Defaults to True.
        window_size: Target window size for centering. Defaults to 196,608.
        batch_size: size of batch for processing embeddings. Defaults to 8.
        save_path: Optional path to save embeddings in compressed npz format.
                  If provided, saves embeddings and IDs as np.savez_compressed.
                  Defaults to None.
        mean_pool: If True, apply mean pooling across the embedding dimension.
                  Defaults to False.

    Returns:
        Tuple containing:
            - embeddings: NumPy array of embeddings (num_sequences, target_length, embedding_dim)
                         or (num_sequences, target_length) if mean_pool=True.
            - sequence_ids: Optional list of sequence IDs if return_sequence_ids=True, else None.
    """
    # Read sequences from FASTA and convert to tensors
    print("Reading fasta file...")
    sequences = read_fasta_sequences(fasta_path)

    # Retrieve embeddings
    print("Retrieving embeddings...")
    sequence_ids, embeddings = retrieve_embeddings(
        sequences, 
        batch_size=batch_size, 
        mean_pool=mean_pool, 
        center_sequences=center_sequences,
        window_size=window_size, 
        pad_value=-1, 
    )

    # Save to npz if save_path is provided
    if save_path is not None:
        # Convert sequence_ids to numpy array
        ids_np = np.array(sequence_ids)
        # Save in compressed npz format
        np.savez_compressed(save_path, ids=ids_np, embeddings=embeddings)

    return embeddings, sequence_ids


if __name__ == "__main__":
    # When run as script, delegate to CLI module
    from retrieve_embeddings.cli import main

    main()
