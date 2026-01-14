"""
Command-line interface for retrieving embeddings from FASTA files.
"""

import argparse
import sys
from pathlib import Path

from retrieve_embeddings.retrieve_embeddings import retrieve_embeddings_from_fasta


def main() -> None:
    """
    Command-line interface for retrieving embeddings from FASTA files.

    Example:
        python -m retrieve_embeddings.cli \\
            --input-file test.fasta \\
            --output-file output/embeddings.npz
    """
    parser = argparse.ArgumentParser(
        description="Retrieve embeddings from FASTA sequences using Enformer model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input FASTA file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output npz file for saving embeddings.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=196_608,
        help="Window size for sequence centering. Defaults to 196608.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing embeddings",
    )
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Disable sequence centering (sequences must be exactly window_size).",
    )
    parser.add_argument(
        "--no-mean-pool",
        dest="mean_pool",
        action="store_false",
        help="Disable mean pooling (default: enabled).",
    )
    
    parser.set_defaults(mean_pool=True)

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process FASTA file and save embeddings
    try:
        print(f"Reading sequences from: {args.input_file}")
        
        embeddings, sequence_ids = retrieve_embeddings_from_fasta(
            fasta_path=args.input_file,
            center_sequences=not args.no_center,
            window_size=args.window_size,
            save_path=args.output_file,
            batch_size=args.batch_size,
            mean_pool=args.mean_pool,
        )

        print(f"Successfully processed {len(sequence_ids)} sequences")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Saved embeddings to: {args.output_file}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
