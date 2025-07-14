#!/usr/bin/env python3
"""
Test parallel processing with just 100 sequences to verify it works.
"""

import pandas as pd
import torch
import torch.multiprocessing as mp
import time
from pathlib import Path
from enformer_pytorch import Enformer, seq_indices_to_one_hot
from embeddings.sequence_embedder import center_sequence_in_window, sequence_to_tensor

def process_chunk_on_gpu(gpu_id, chunk_data, chunk_idx, output_dir, batch_size=4):
    """Process a chunk of data on a specific GPU."""
    try:
        # Set device
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        
        print(f"Device {device}: Processing chunk {chunk_idx} with {len(chunk_data)} sequences")
        
        # Create model on this device
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
                
                # Stack and move to device
                batch_tensor = torch.stack(batch_tensors).to(device)
                one_hot = seq_indices_to_one_hot(batch_tensor)
                
                # Get embeddings
                _, embeddings = model(one_hot, return_embeddings=True)
                averaged_embeddings = embeddings.mean(dim=-1).cpu()
                
                embeddings_list.append(averaged_embeddings)
                variant_ids.extend(batch_variant_ids)
                
                print(f"Device {device}: Processed batch {i//batch_size + 1} in chunk {chunk_idx}")
        
        # Combine embeddings for this chunk
        chunk_embeddings = torch.cat(embeddings_list, dim=0)
        
        # Save chunk result
        chunk_output_path = output_dir / f"test_chunk_{chunk_idx}_device_{gpu_id}.pt"
        torch.save({
            'embeddings': chunk_embeddings,
            'variant_ids': variant_ids,
            'chunk_idx': chunk_idx,
            'device': str(device),
            'shape': chunk_embeddings.shape
        }, chunk_output_path)
        
        print(f"Device {device}: Completed chunk {chunk_idx}, shape: {chunk_embeddings.shape}")
        return chunk_output_path
        
    except Exception as e:
        print(f"Device {device}: Error processing chunk {chunk_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Configuration
    CSV_PATH = "/lambda/nfs/zelun-enformer/enformer-embeddings/data/166k_rice/post_embeddings/all_data_with_sequence.csv"
    OUTPUT_DIR = Path("/lambda/nfs/zelun-enformer/enformer-embeddings/embeddings/test_parallel")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    BATCH_SIZE = 4  # Smaller batches for testing
    TEST_SEQUENCES = 100  # Only process 100 sequences
    
    print("=" * 70)
    print("TEST PARALLEL EMBEDDINGS PROCESSING (100 sequences)")
    print("=" * 70)
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus == 0:
        print("No GPUs available, using CPU")
        num_devices = 1
    else:
        num_devices = num_gpus
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Load test data
    print(f"\\nLoading first {TEST_SEQUENCES} sequences...")
    df = pd.read_csv(CSV_PATH, nrows=TEST_SEQUENCES)
    total_sequences = len(df)
    print(f"Test sequences: {total_sequences}")
    
    # Split data into chunks for each device
    chunk_size = max(1, total_sequences // num_devices)
    chunks = []
    
    for i in range(num_devices):
        start_idx = i * chunk_size
        if i == num_devices - 1:  # Last device gets remaining sequences
            end_idx = total_sequences
        else:
            end_idx = min((i + 1) * chunk_size, total_sequences)
        
        if start_idx < total_sequences:
            chunk_df = df.iloc[start_idx:end_idx]
            chunks.append((i, chunk_df, i))
            print(f"Device {i}: {len(chunk_df)} sequences (rows {start_idx}-{end_idx})")
    
    # Start processing
    print("\\nStarting test processing...")
    start_time = time.time()
    
    if num_devices == 1:
        # Single device processing
        results = []
        for gpu_id, chunk_data, chunk_idx in chunks:
            result = process_chunk_on_gpu(gpu_id, chunk_data, chunk_idx, OUTPUT_DIR, BATCH_SIZE)
            results.append(result)
    else:
        # Multi-GPU processing
        mp.set_start_method('spawn', force=True)
        
        with mp.Pool(num_devices) as pool:
            args = [(gpu_id, chunk_data, chunk_idx, OUTPUT_DIR, BATCH_SIZE) 
                    for gpu_id, chunk_data, chunk_idx in chunks]
            results = pool.starmap(process_chunk_on_gpu, args)
    
    # Combine results
    print("\\nCombining results...")
    all_embeddings = []
    all_variant_ids = []
    
    for chunk_idx, result_path in enumerate(results):
        if result_path and result_path.exists():
            chunk_data = torch.load(result_path)
            all_embeddings.append(chunk_data['embeddings'])
            all_variant_ids.extend(chunk_data['variant_ids'])
            print(f"Loaded chunk {chunk_idx}: {chunk_data['shape']}")
    
    # Final combination
    if all_embeddings:
        final_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Save test result
        final_output_path = OUTPUT_DIR / "test_embeddings_100.pt"
        torch.save({
            'embeddings': final_embeddings,
            'variant_ids': all_variant_ids,
            'metadata': {
                'total_sequences': len(all_variant_ids),
                'embedding_shape': final_embeddings.shape,
                'processing_time_seconds': time.time() - start_time,
                'num_devices_used': num_devices,
                'test_run': True
            }
        }, final_output_path)
        
        total_time = time.time() - start_time
        print("\\n" + "=" * 70)
        print("TEST PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"Sequences processed: {len(all_variant_ids)}")
        print(f"Final embeddings shape: {final_embeddings.shape}")
        print(f"Processing time: {total_time:.2f} seconds")
        print(f"Time per sequence: {total_time/len(all_variant_ids):.3f} seconds")
        print(f"Devices used: {num_devices}")
        print(f"Output saved to: {final_output_path}")
        
        # Cleanup
        for result_path in results:
            if result_path and result_path.exists():
                result_path.unlink()
        
        print("\\nTest successful! Ready for full processing.")
    else:
        print("No results generated - check for errors above")

if __name__ == "__main__":
    main()