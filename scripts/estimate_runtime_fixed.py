#!/usr/bin/env python3
"""
Estimate processing time for all 121k sequences based on actual timing tests.
Fixed version that handles sequence length properly.
"""

import pandas as pd
import torch
import time
from enformer_pytorch import Enformer, seq_indices_to_one_hot

def sequence_to_tensor(sequence):
    """Convert DNA sequence to tensor representation. A=0, C=1, G=2, T=3, N=4"""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    indices = [mapping.get(base.upper(), 4) for base in sequence]
    return torch.tensor(indices, dtype=torch.long)

def time_batch_processing(sequences, model, batch_size=8):
    """Time the processing of a batch of sequences."""
    start_time = time.time()
    
    with torch.no_grad():
        # Convert sequences to tensors
        batch_tensors = []
        for seq in sequences:
            tensor_seq = sequence_to_tensor(seq)
            batch_tensors.append(tensor_seq)
        
        # Stack into batch
        batch_tensor = torch.stack(batch_tensors)
        
        # Convert to one-hot
        one_hot = seq_indices_to_one_hot(batch_tensor)
        
        # Get embeddings from model
        _, embeddings = model(one_hot, return_embeddings=True)
        
        # Average across the feature dimension (3072) to get (batch, 896)
        averaged_embeddings = embeddings.mean(dim=-1)
    
    end_time = time.time()
    return end_time - start_time, averaged_embeddings.shape

def main():
    print("=" * 60)
    print("RUNTIME ESTIMATION FOR ENFORMER EMBEDDINGS")
    print("=" * 60)
    
    # Create model from scratch to avoid TF gamma issues
    print("Creating Enformer model...")
    start_time = time.time()
    model = Enformer.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        output_heads=dict(human=5313, mouse=1643),
        target_length=896,
        use_tf_gamma=False  # Disable TF gamma to avoid length restrictions
    )
    model.eval()
    model_load_time = time.time() - start_time
    print(f"Model created in {model_load_time:.2f} seconds")
    
    # Load sample data
    print("Loading sample data...")
    CSV_PATH = "/lambda/nfs/zelun-enformer/enformer-embeddings/data/166k_rice/post_embeddings/all_data_with_sequence.csv"
    df = pd.read_csv(CSV_PATH, nrows=20)  # Load first 20 sequences for timing
    total_sequences = 121292  # From our earlier count
    
    # Test single sequence first
    print("\\nTesting single sequence...")
    test_sequence = df['Sequence'].iloc[0]
    print(f"Sequence length: {len(test_sequence)}")
    
    single_time, single_shape = time_batch_processing([test_sequence], model, 1)
    print(f"Single sequence processing time: {single_time:.3f}s")
    print(f"Output shape: {single_shape}")
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8]
    timing_results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(df):
            continue
            
        print(f"\\nTesting batch size {batch_size}...")
        
        # Test multiple times for accuracy
        times = []
        for i in range(2):  # 2 test runs
            test_sequences = df['Sequence'].iloc[:batch_size].tolist()
            processing_time, output_shape = time_batch_processing(test_sequences, model, batch_size)
            times.append(processing_time)
            print(f"  Run {i+1}: {processing_time:.3f}s")
        
        avg_time = sum(times) / len(times)
        time_per_sequence = avg_time / batch_size
        
        timing_results[batch_size] = {
            'avg_time': avg_time,
            'time_per_sequence': time_per_sequence,
            'shape': output_shape
        }
        
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Time per sequence: {time_per_sequence:.3f}s")
        print(f"  Output shape: {output_shape}")
    
    # Calculate estimates for different batch sizes
    print("\\n" + "=" * 60)
    print("PROCESSING TIME ESTIMATES")
    print("=" * 60)
    
    for batch_size, results in timing_results.items():
        total_batches = (total_sequences + batch_size - 1) // batch_size
        total_time_seconds = total_batches * results['avg_time']
        total_time_hours = total_time_seconds / 3600
        total_time_days = total_time_hours / 24
        
        print(f"\\nBatch size {batch_size}:")
        print(f"  Total batches needed: {total_batches:,}")
        print(f"  Estimated total time: {total_time_hours:.1f} hours ({total_time_days:.1f} days)")
        print(f"  Memory usage: ~{batch_size * 196608 * 4 / 1024**2:.1f} MB per batch")
    
    # Recommended configuration
    print("\\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    # Find optimal batch size (balance between speed and memory)
    recommended_batch_size = 8 if 8 in timing_results else max(timing_results.keys())
    if recommended_batch_size in timing_results:
        results = timing_results[recommended_batch_size]
        total_batches = (total_sequences + recommended_batch_size - 1) // recommended_batch_size
        total_time_seconds = total_batches * results['avg_time']
        total_time_hours = total_time_seconds / 3600
        total_time_days = total_time_hours / 24
        
        print(f"Recommended batch size: {recommended_batch_size}")
        print(f"Expected processing time: {total_time_hours:.1f} hours ({total_time_days:.1f} days)")
        print(f"Expected memory usage: ~{recommended_batch_size * 196608 * 4 / 1024**2:.1f} MB per batch")
        print(f"Output tensor size: {total_sequences} x 896 = {total_sequences * 896 * 4 / 1024**3:.2f} GB")
    
    print(f"\\nNote: This uses a fresh model (not pretrained weights)")
    print(f"For pretrained weights, you may need to adjust sequence length or use different approach")

if __name__ == "__main__":
    main()