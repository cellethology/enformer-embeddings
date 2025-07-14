#!/usr/bin/env python3
"""
Production script to process all 121k rice sequences and extract averaged embeddings.
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
from enformer_pytorch import from_pretrained, seq_indices_to_one_hot

def sequence_to_tensor(sequence):
    """Convert DNA sequence to tensor representation. A=0, C=1, G=2, T=3, N=4"""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    indices = [mapping.get(base.upper(), 4) for base in sequence]
    return torch.tensor(indices, dtype=torch.long)

def process_embeddings_batch(sequences, model, batch_size=8):
    """Process a batch of sequences and return averaged embeddings."""
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            # Convert sequences to tensors
            batch_tensors = []
            for seq in batch_sequences:
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
            
            embeddings_list.append(averaged_embeddings)
    
    # Concatenate all batches
    return torch.cat(embeddings_list, dim=0)

def main():
    # Configuration
    CSV_PATH = "/lambda/nfs/zelun-enformer/enformer-embeddings/data/166k_rice/post_embeddings/all_data_with_sequence.csv"
    OUTPUT_PATH = "/lambda/nfs/zelun-enformer/enformer-embeddings/embeddings/all_averaged_embeddings.pt"
    BATCH_SIZE = 8  # Process 8 sequences at once
    CHUNK_SIZE = 1000  # Process 1000 sequences per chunk to manage memory
    
    print("=" * 60)
    print("ENFORMER EMBEDDINGS PROCESSING")
    print("=" * 60)
    
    # Load model
    print("Loading pretrained Enformer model...")
    start_time = time.time()
    model = from_pretrained('EleutherAI/enformer-official-rough')
    model.eval()
    model_load_time = time.time() - start_time
    print(f"Model loaded in {model_load_time:.2f} seconds")
    
    # Load data
    print("Loading CSV data...")
    df = pd.read_csv(CSV_PATH)
    total_sequences = len(df)
    print(f"Total sequences to process: {total_sequences:,}")
    
    # Estimate processing time
    # Based on our test: ~2-3 seconds per batch of 8 sequences
    estimated_batches = (total_sequences + BATCH_SIZE - 1) // BATCH_SIZE
    estimated_time_minutes = (estimated_batches * 2.5) / 60  # 2.5 seconds per batch
    estimated_time_hours = estimated_time_minutes / 60
    
    print(f"Estimated processing time: {estimated_time_hours:.1f} hours ({estimated_time_minutes:.0f} minutes)")
    print(f"Estimated batches: {estimated_batches:,}")
    
    # Initialize storage
    all_embeddings = []
    processed_count = 0
    
    # Process in chunks
    print("\\nStarting processing...")
    overall_start = time.time()
    
    for chunk_start in range(0, total_sequences, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_sequences)
        chunk_df = df.iloc[chunk_start:chunk_end]
        
        print(f"\\nProcessing chunk {chunk_start//CHUNK_SIZE + 1}/{(total_sequences + CHUNK_SIZE - 1)//CHUNK_SIZE}")
        print(f"Sequences {chunk_start:,} to {chunk_end:,}")
        
        # Process this chunk
        sequences = chunk_df['Sequence'].tolist()
        
        chunk_start_time = time.time()
        chunk_embeddings = process_embeddings_batch(sequences, model, BATCH_SIZE)
        chunk_time = time.time() - chunk_start_time
        
        all_embeddings.append(chunk_embeddings)
        processed_count += len(sequences)
        
        # Progress update
        progress = (processed_count / total_sequences) * 100
        elapsed_time = time.time() - overall_start
        estimated_total_time = elapsed_time * (total_sequences / processed_count)
        remaining_time = estimated_total_time - elapsed_time
        
        print(f"Chunk processed in {chunk_time:.2f}s ({chunk_time/len(sequences):.3f}s per sequence)")
        print(f"Progress: {progress:.1f}% ({processed_count:,}/{total_sequences:,})")
        print(f"Elapsed: {elapsed_time/3600:.2f}h, Remaining: {remaining_time/3600:.2f}h")
        
        # Save intermediate results every 10 chunks
        if (chunk_start // CHUNK_SIZE + 1) % 10 == 0:
            print("Saving intermediate results...")
            intermediate_embeddings = torch.cat(all_embeddings, dim=0)
            torch.save(intermediate_embeddings, OUTPUT_PATH.replace('.pt', f'_intermediate_{processed_count}.pt'))
    
    # Combine all embeddings
    print("\\nCombining all embeddings...")
    final_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Save final results
    print(f"Saving final embeddings to {OUTPUT_PATH}")
    torch.save(final_embeddings, OUTPUT_PATH)
    
    total_time = time.time() - overall_start
    print("\\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Total sequences processed: {processed_count:,}")
    print(f"Final embeddings shape: {final_embeddings.shape}")
    print(f"Total processing time: {total_time/3600:.2f} hours")
    print(f"Average time per sequence: {total_time/processed_count:.3f} seconds")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()