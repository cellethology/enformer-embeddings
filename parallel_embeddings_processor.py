#!/usr/bin/env python3
"""
Parallelized embeddings processor that uses all available GPUs.
"""

import pandas as pd
import torch
import torch.multiprocessing as mp
import time
from pathlib import Path
from enformer_pytorch import Enformer, seq_indices_to_one_hot
from embeddings.sequence_embedder import center_sequence_in_window, sequence_to_tensor

def process_chunk_on_gpu(gpu_id, chunk_data, chunk_idx, output_dir, batch_size=8):
    """Process a chunk of data on a specific GPU."""
    try:
        # Set device
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        
        print(f"GPU {gpu_id}: Processing chunk {chunk_idx} with {len(chunk_data)} sequences")
        
        # Create model on this GPU
        model = Enformer.from_hparams(
            dim=1536,
            depth=11,
            heads=8,
            output_heads=dict(human=5313, mouse=1643),
            target_length=896,
            use_tf_gamma=False
        ).to(device)
        model.eval()
        
        embeddings_list = []
        variant_ids = []
        
        # Process in batches
        with torch.no_grad():
            for i in range(0, len(chunk_data), batch_size):
                batch_df = chunk_data.iloc[i:i+batch_size]
                batch_sequences = batch_df['Sequence'].tolist()
                batch_variant_ids = batch_df['Variant_ID'].tolist()
                
                # Convert sequences to tensors
                batch_tensors = []
                for seq in batch_sequences:
                    centered_seq = center_sequence_in_window(seq, 196608)
                    tensor_seq = sequence_to_tensor(centered_seq)
                    batch_tensors.append(tensor_seq)
                
                # Stack and move to GPU
                batch_tensor = torch.stack(batch_tensors).to(device)
                one_hot = seq_indices_to_one_hot(batch_tensor)
                
                # Get embeddings
                _, embeddings = model(one_hot, return_embeddings=True)
                averaged_embeddings = embeddings.mean(dim=-1).cpu()
                
                embeddings_list.append(averaged_embeddings)
                variant_ids.extend(batch_variant_ids)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"GPU {gpu_id}: Processed {i + len(batch_sequences)} sequences in chunk {chunk_idx}")
        
        # Combine embeddings for this chunk
        chunk_embeddings = torch.cat(embeddings_list, dim=0)
        
        # Save chunk result
        chunk_output_path = output_dir / f"chunk_{chunk_idx}_gpu_{gpu_id}.pt"
        torch.save({
            'embeddings': chunk_embeddings,
            'variant_ids': variant_ids,
            'chunk_idx': chunk_idx,
            'gpu_id': gpu_id,
            'shape': chunk_embeddings.shape
        }, chunk_output_path)
        
        print(f"GPU {gpu_id}: Completed chunk {chunk_idx}, saved to {chunk_output_path}")
        return chunk_output_path
        
    except Exception as e:
        print(f"GPU {gpu_id}: Error processing chunk {chunk_idx}: {e}")
        return None

def main():
    # Configuration
    CSV_PATH = "/lambda/nfs/zelun-enformer/enformer-embeddings/data/166k_rice/post_embeddings/all_data_with_sequence.csv"
    OUTPUT_DIR = Path("/lambda/nfs/zelun-enformer/enformer-embeddings/embeddings/parallel_output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    BATCH_SIZE = 8  # Per GPU batch size
    
    print("=" * 70)
    print("PARALLEL ENFORMER EMBEDDINGS PROCESSING")
    print("=" * 70)
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus == 0:
        print("No GPUs available, falling back to CPU processing")
        # Could implement CPU fallback here
        return
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Load data
    print("\\nLoading data...")
    df = pd.read_csv(CSV_PATH)
    total_sequences = len(df)
    print(f"Total sequences: {total_sequences:,}")
    
    # Split data into chunks for each GPU
    chunk_size = total_sequences // num_gpus
    chunks = []
    
    for i in range(num_gpus):
        start_idx = i * chunk_size
        if i == num_gpus - 1:  # Last GPU gets remaining sequences
            end_idx = total_sequences
        else:
            end_idx = (i + 1) * chunk_size
        
        chunk_df = df.iloc[start_idx:end_idx]
        chunks.append((i, chunk_df, i))  # (gpu_id, data, chunk_idx)
        print(f"GPU {i}: {len(chunk_df):,} sequences (rows {start_idx:,}-{end_idx:,})")
    
    # Estimate processing time
    sequences_per_gpu = total_sequences // num_gpus
    estimated_batches_per_gpu = (sequences_per_gpu + BATCH_SIZE - 1) // BATCH_SIZE
    estimated_time_per_gpu = estimated_batches_per_gpu * 2.0 / 3600  # 2 seconds per batch
    
    print(f"\\nEstimated time per GPU: {estimated_time_per_gpu:.1f} hours")
    print(f"Estimated total time (parallel): {estimated_time_per_gpu:.1f} hours")
    print(f"Speedup vs sequential: {num_gpus}x")
    
    # Start parallel processing
    print("\\nStarting parallel processing...")
    start_time = time.time()
    
    # Use multiprocessing to run on multiple GPUs
    mp.set_start_method('spawn', force=True)
    
    with mp.Pool(num_gpus) as pool:
        # Create arguments for each process
        args = [(gpu_id, chunk_data, chunk_idx, OUTPUT_DIR, BATCH_SIZE) 
                for gpu_id, chunk_data, chunk_idx in chunks]
        
        # Process chunks in parallel
        results = pool.starmap(process_chunk_on_gpu, args)
    
    # Combine results
    print("\\nCombining results from all GPUs...")
    all_embeddings = []
    all_variant_ids = []
    
    for chunk_idx, result_path in enumerate(results):
        if result_path and result_path.exists():
            chunk_data = torch.load(result_path)
            all_embeddings.append(chunk_data['embeddings'])
            all_variant_ids.extend(chunk_data['variant_ids'])
            print(f"Loaded chunk {chunk_idx}: {chunk_data['shape']}")
    
    # Final combination
    final_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Save final result
    final_output_path = OUTPUT_DIR / "all_embeddings_parallel.pt"
    torch.save({
        'embeddings': final_embeddings,
        'variant_ids': all_variant_ids,
        'metadata': {
            'total_sequences': len(all_variant_ids),
            'embedding_shape': final_embeddings.shape,
            'processing_time_hours': (time.time() - start_time) / 3600,
            'num_gpus_used': num_gpus,
            'chunks_processed': len(results)
        }
    }, final_output_path)
    
    total_time = time.time() - start_time
    print("\\n" + "=" * 70)
    print("PARALLEL PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Total sequences processed: {len(all_variant_ids):,}")
    print(f"Final embeddings shape: {final_embeddings.shape}")
    print(f"Processing time: {total_time/3600:.2f} hours")
    print(f"GPUs used: {num_gpus}")
    print(f"Speedup achieved: ~{num_gpus}x")
    print(f"Average time per sequence: {total_time/len(all_variant_ids):.4f} seconds")
    print(f"Output saved to: {final_output_path}")
    
    # Cleanup chunk files
    print("\\nCleaning up temporary chunk files...")
    for result_path in results:
        if result_path and result_path.exists():
            result_path.unlink()
    
    print("\\nTo load results:")
    print("import torch")
    print(f"data = torch.load('{final_output_path}')")
    print("embeddings = data['embeddings']")
    print("variant_ids = data['variant_ids']")

if __name__ == "__main__":
    main()