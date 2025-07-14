#!/usr/bin/env python3
"""
Final production script to process all 121k rice sequences and extract averaged embeddings.
Uses the existing sequence_embedder.py functions with proper sequence centering.
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
from enformer_pytorch import Enformer, seq_indices_to_one_hot
from embeddings.sequence_embedder import center_sequence_in_window, sequence_to_tensor

def process_embeddings_batch(sequences, model, batch_size=8):
    """Process a batch of sequences and return averaged embeddings."""
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            # Center sequences in 196,608 window and convert to tensors
            batch_tensors = []
            for seq in batch_sequences:
                # Center the sequence in the required window
                centered_seq = center_sequence_in_window(seq, 196608)
                tensor_seq = sequence_to_tensor(centered_seq)
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
    OUTPUT_DIR = Path("/lambda/nfs/zelun-enformer/enformer-embeddings/embeddings")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    BATCH_SIZE = 8  # Process 8 sequences at once
    CHUNK_SIZE = 1000  # Process 1000 sequences per chunk to manage memory
    
    print("=" * 70)
    print("ENFORMER EMBEDDINGS PROCESSING - FINAL VERSION")
    print("=" * 70)
    
    # Create model (using fresh model to avoid pretrained restrictions)
    print("Creating Enformer model...")
    start_time = time.time()
    model = Enformer.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        output_heads=dict(human=5313, mouse=1643),
        target_length=896,
        use_tf_gamma=False
    )
    model.eval()
    model_load_time = time.time() - start_time
    print(f"Model created in {model_load_time:.2f} seconds")
    
    # Load data
    print("Loading CSV data...")
    df = pd.read_csv(CSV_PATH)
    total_sequences = len(df)
    print(f"Total sequences to process: {total_sequences:,}")
    
    # Check sequence lengths
    sample_lengths = [len(seq) for seq in df['Sequence'].head(10)]
    print(f"Sample sequence lengths: {sample_lengths}")
    print(f"Will be centered in 196,608 windows with N-padding")
    
    # Time estimates
    estimated_batches = (total_sequences + BATCH_SIZE - 1) // BATCH_SIZE
    estimated_time_hours = estimated_batches * 2.0 / 3600  # 2 seconds per batch estimate
    
    print(f"Estimated processing time: {estimated_time_hours:.1f} hours")
    print(f"Estimated batches: {estimated_batches:,}")
    
    # Initialize storage
    all_embeddings = []
    all_variant_ids = []
    all_expressions = []
    processed_count = 0
    
    # Process in chunks
    print("\\nStarting processing...")
    overall_start = time.time()
    
    total_chunks = (total_sequences + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, total_sequences)
        chunk_df = df.iloc[chunk_start:chunk_end]
        
        print(f"\\nProcessing chunk {chunk_idx + 1}/{total_chunks}")
        print(f"Sequences {chunk_start:,} to {chunk_end:,}")
        
        # Process this chunk
        sequences = chunk_df['Sequence'].tolist()
        variant_ids = chunk_df['Variant_ID'].tolist()
        expressions = chunk_df['Expression'].tolist()
        chunk_start_time = time.time()
        chunk_embeddings = process_embeddings_batch(sequences, model, BATCH_SIZE)
        chunk_time = time.time() - chunk_start_time
        
        all_embeddings.append(chunk_embeddings)
        all_variant_ids.extend(variant_ids)
        all_expressions.extend(expressions)
        processed_count += len(sequences)
        
        # Progress update
        progress = (processed_count / total_sequences) * 100
        elapsed_time = time.time() - overall_start
        
        if processed_count > 0:
            estimated_total_time = elapsed_time * (total_sequences / processed_count)
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0
        
        print(f"Chunk processed in {chunk_time:.2f}s ({chunk_time/len(sequences):.3f}s per sequence)")
        print(f"Progress: {progress:.1f}% ({processed_count:,}/{total_sequences:,})")
        print(f"Elapsed: {elapsed_time/3600:.2f}h, Remaining: {remaining_time/3600:.2f}h")
        
        # Save intermediate results every 10 chunks
        if (chunk_idx + 1) % 10 == 0:
            print("Saving intermediate results...")
            intermediate_embeddings = torch.cat(all_embeddings, dim=0)
            intermediate_path = OUTPUT_DIR / f"embeddings_intermediate_{processed_count}.pt"
            torch.save({
                'embeddings': intermediate_embeddings,
                'variant_ids': all_variant_ids[:len(intermediate_embeddings)],
                'expressions': all_expressions[:len(intermediate_embeddings)],
                'processed_count': processed_count
            }, intermediate_path)
    
    # Combine all embeddings
    print("\\nCombining all embeddings...")
    final_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Save final results
    final_output_path = OUTPUT_DIR / "all_averaged_embeddings.pt"
    print(f"Saving final embeddings to {final_output_path}")
    
    torch.save({
        'embeddings': final_embeddings,
        'expressions': all_expressions,
        'variant_ids': all_variant_ids,
        'metadata': {
            'total_sequences': processed_count,
            'embedding_shape': final_embeddings.shape,
            'processing_time_hours': (time.time() - overall_start) / 3600,
            'model_config': {
                'dim': 1536,
                'depth': 11,
                'heads': 8,
                'target_length': 896,
                'use_tf_gamma': False
            }
        }
    }, final_output_path)
    
    # Also save as CSV for easy access
    csv_output_path = OUTPUT_DIR / "embeddings_summary.csv"
    summary_df = pd.DataFrame({
        'Variant_ID': all_variant_ids,
        'Embedding_Shape': [final_embeddings.shape[1]] * len(all_variant_ids)
    })
    summary_df.to_csv(csv_output_path, index=False)
    
    total_time = time.time() - overall_start
    print("\\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Total sequences processed: {processed_count:,}")
    print(f"Final embeddings shape: {final_embeddings.shape}")
    print(f"Total processing time: {total_time/3600:.2f} hours")
    print(f"Average time per sequence: {total_time/processed_count:.3f} seconds")
    print(f"Main output: {final_output_path}")
    print(f"Summary CSV: {csv_output_path}")
    
    print("\\nTo load the results:")
    print("import torch")
    print(f"data = torch.load('{final_output_path}')")
    print("embeddings = data['embeddings']  # Shape: (121292, 896)")
    print("variant_ids = data['variant_ids']")

if __name__ == "__main__":
    main()