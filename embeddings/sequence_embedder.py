import pandas as pd
import torch
import numpy as np
from pathlib import Path
from enformer_pytorch import from_pretrained, seq_indices_to_one_hot

def center_sequence_in_window(sequence, window_size=196608):
    """
    Center a sequence in a window of specified size, padding with 'N' characters.
    
    Args:
        sequence (str): Input DNA sequence
        window_size (int): Target window size (default: 196608)
    
    Returns:
        str: Centered sequence padded with 'N' characters
    """
    seq_len = len(sequence)
    
    if seq_len >= window_size:
        # If sequence is longer than window, truncate from center
        start = (seq_len - window_size) // 2
        return sequence[start:start + window_size]
    
    # Calculate padding needed
    pad_total = window_size - seq_len
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    
    # Pad with 'N' characters
    centered_sequence = 'N' * pad_left + sequence + 'N' * pad_right
    
    return centered_sequence

def process_sequences_from_csv(csv_path, output_path=None):
    """
    Process sequences from CSV file, centering each in 196,608 window.
    
    Args:
        csv_path (str): Path to input CSV file
        output_path (str): Path to output CSV file (optional)
    
    Returns:
        pd.DataFrame: DataFrame with processed sequences
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Process each sequence
    df['Processed_Sequence'] = df['Sequence'].apply(center_sequence_in_window)
    
    # Verify all sequences are exactly 196,608 characters
    df['Sequence_Length'] = df['Processed_Sequence'].apply(len)
    
    print(f"Processing complete. All sequences are length {df['Sequence_Length'].iloc[0]}")
    print(f"Total sequences processed: {len(df)}")
    
    # Save to output file if specified
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    
    return df

def sequence_to_tensor(sequence):
    """
    Convert DNA sequence to tensor representation.
    A=0, C=1, G=2, T=3, N=4
    
    Args:
        sequence (str): DNA sequence
    
    Returns:
        torch.Tensor: Tensor representation of sequence
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    # Convert sequence to indices
    indices = [mapping.get(base.upper(), 4) for base in sequence]
    
    return torch.tensor(indices, dtype=torch.long)

def get_sequence_embeddings(sequences, model=None, batch_size=1):
    """
    Get averaged embeddings for DNA sequences using Enformer model.
    
    Args:
        sequences (list): List of DNA sequences
        model: Pretrained Enformer model (loads if None)
        batch_size (int): Batch size for processing
    
    Returns:
        torch.Tensor: Averaged embeddings of shape (batch, 896)
    """
    if model is None:
        print("Loading pretrained Enformer model...")
        model = from_pretrained('EleutherAI/enformer-official-rough')
        model.eval()
    
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
            
            print(f"Processed batch {i//batch_size + 1}, embeddings shape: {averaged_embeddings.shape}")
    
    # Concatenate all batches
    all_embeddings = torch.cat(embeddings_list, dim=0)
    
    return all_embeddings

def main():
    # Set paths
    csv_path = "/home/ubuntu/zelun-enformer/enformer-embeddings/data/166k_rice/post_embeddings/all_data_with_sequence.csv"
    output_path = "/lambda/nfs/zelun-enformer/enformer-embeddings/embeddings/processed_sequences.csv"
    
    # Process sequences
    df = process_sequences_from_csv(csv_path, output_path)
    
    # Example: Get embeddings for first few sequences
    print("Getting embeddings for first 3 sequences...")
    first_sequences = df['Processed_Sequence'].iloc[:3].tolist()
    
    # Get averaged embeddings
    embeddings = get_sequence_embeddings(first_sequences, batch_size=1)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First embedding sample: {embeddings[0][:10]}")
    
    return df, embeddings

if __name__ == "__main__":
    main()