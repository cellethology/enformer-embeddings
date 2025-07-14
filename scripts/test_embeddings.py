#!/usr/bin/env python3
"""Test script for getting averaged embeddings from sequences."""

import torch
from enformer_pytorch import from_pretrained, seq_indices_to_one_hot

def sequence_to_tensor(sequence):
    """Convert DNA sequence to tensor representation. A=0, C=1, G=2, T=3, N=4"""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    indices = [mapping.get(base.upper(), 4) for base in sequence]
    return torch.tensor(indices, dtype=torch.long)

def get_averaged_embeddings(sequences, model=None):
    """Get averaged embeddings (batch, 896) from sequences."""
    if model is None:
        print("Loading pretrained Enformer model...")
        model = from_pretrained('EleutherAI/enformer-official-rough')
        model.eval()
    
    # Convert sequences to tensors
    batch_tensors = []
    for seq in sequences:
        tensor_seq = sequence_to_tensor(seq)
        batch_tensors.append(tensor_seq)
    
    # Stack into batch
    batch_tensor = torch.stack(batch_tensors)
    print(f"Batch tensor shape: {batch_tensor.shape}")
    
    # Convert to one-hot
    one_hot = seq_indices_to_one_hot(batch_tensor)
    print(f"One-hot shape: {one_hot.shape}")
    
    with torch.no_grad():
        # Get embeddings from model
        _, embeddings = model(one_hot, return_embeddings=True)
        print(f"Original embeddings shape: {embeddings.shape}")
        
        # Average across the feature dimension (3072) to get (batch, 896)
        averaged_embeddings = embeddings.mean(dim=-1)
        print(f"Averaged embeddings shape: {averaged_embeddings.shape}")
    
    return averaged_embeddings

def main():
    # Test with simple sequences
    test_sequences = [
        "ATCGANCG" * 24576,  # 196608 length
        "GCTAGCTA" * 24576,  # 196608 length
    ]
    
    print(f"Testing with {len(test_sequences)} sequences of length {len(test_sequences[0])}")
    
    # Get averaged embeddings
    embeddings = get_averaged_embeddings(test_sequences)
    
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Sample values from first embedding: {embeddings[0][:10]}")
    
    return embeddings

if __name__ == "__main__":
    main()