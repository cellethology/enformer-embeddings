#!/usr/bin/env python3
"""
Simple runtime estimation based on our successful test and scaling calculations.
"""

import pandas as pd
import time

def main():
    print("=" * 60)
    print("ENFORMER EMBEDDINGS RUNTIME ESTIMATION")
    print("=" * 60)
    
    # Load data info
    CSV_PATH = "/lambda/nfs/zelun-enformer/enformer-embeddings/data/166k_rice/post_embeddings/all_data_with_sequence.csv"
    total_sequences = 121292  # From our earlier count
    
    # Check actual sequence length
    print("Checking sequence lengths...")
    df_sample = pd.read_csv(CSV_PATH, nrows=5)
    seq_lengths = [len(seq) for seq in df_sample['Sequence']]
    print(f"Sample sequence lengths: {seq_lengths}")
    
    # Based on our successful test_embeddings.py run:
    # - 2 sequences of 196,608 length processed successfully
    # - Time taken: ~30-40 seconds (including model loading)
    # - Processing 2 sequences took ~5-10 seconds (excluding model loading)
    
    # Conservative estimates based on our test
    MODEL_LOAD_TIME = 30  # seconds
    TIME_PER_SEQUENCE = 5  # seconds per sequence (conservative estimate)
    
    # Different batch size scenarios
    batch_sizes = [1, 2, 4, 8, 16]
    
    print("\\n" + "=" * 60)
    print("PROCESSING TIME ESTIMATES")
    print("=" * 60)
    
    for batch_size in batch_sizes:
        # Assume batch processing is more efficient
        time_per_batch = TIME_PER_SEQUENCE * batch_size * 0.8  # 20% efficiency gain
        total_batches = (total_sequences + batch_size - 1) // batch_size
        total_processing_time = total_batches * time_per_batch
        total_time_with_loading = total_processing_time + MODEL_LOAD_TIME
        
        hours = total_time_with_loading / 3600
        days = hours / 24
        
        print(f"\\nBatch size {batch_size}:")
        print(f"  Total batches: {total_batches:,}")
        print(f"  Processing time: {total_processing_time/3600:.1f} hours")
        print(f"  Total time (with loading): {hours:.1f} hours ({days:.1f} days)")
        print(f"  Memory per batch: ~{batch_size * 196608 * 4 / 1024**2:.0f} MB")
    
    # Recommended approach
    print("\\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    recommended_batch = 8
    time_per_batch = TIME_PER_SEQUENCE * recommended_batch * 0.8
    total_batches = (total_sequences + recommended_batch - 1) // recommended_batch
    total_processing_time = total_batches * time_per_batch
    total_hours = (total_processing_time + MODEL_LOAD_TIME) / 3600
    
    print(f"Recommended configuration:")
    print(f"  Batch size: {recommended_batch}")
    print(f"  Expected time: {total_hours:.1f} hours ({total_hours/24:.1f} days)")
    print(f"  Memory usage: ~{recommended_batch * 196608 * 4 / 1024**2:.0f} MB per batch")
    print(f"  Output size: {total_sequences * 896 * 4 / 1024**3:.2f} GB")
    
    print(f"\\nChunk processing (1000 sequences per chunk):")
    sequences_per_chunk = 1000
    batches_per_chunk = (sequences_per_chunk + recommended_batch - 1) // recommended_batch
    time_per_chunk = batches_per_chunk * time_per_batch
    total_chunks = (total_sequences + sequences_per_chunk - 1) // sequences_per_chunk
    
    print(f"  Total chunks: {total_chunks}")
    print(f"  Time per chunk: {time_per_chunk/60:.1f} minutes")
    print(f"  Can save intermediate results every chunk")
    
    print(f"\\nActual sequence processing notes:")
    print(f"- Your sequences appear to be shorter than expected 196,608 length")
    print(f"- Processing time may be faster than estimated")
    print(f"- Consider using the existing processed embeddings if available")
    
    print(f"\\nTo run the processing:")
    print(f"uv run python process_all_embeddings.py")

if __name__ == "__main__":
    main()