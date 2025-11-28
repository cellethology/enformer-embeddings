"""
Command-line interface for retrieving embeddings from FASTA files.
"""

import argparse
import sys
from pathlib import Path

from retrieve_embeddings.retrieve_embeddings import (
    create_enformer_model,
    retrieve_embeddings_from_fasta,
)


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
        "--pad-value",
        type=str,
        default="N",
        choices=["N", "-", "-1"],
        help="Padding value: 'N', '-', or '-1'. Defaults to 'N'.",
    )
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Disable sequence centering (sequences must be exactly window_size).",
    )
    parser.add_argument(
        "--mean-pool",
        action="store_true",
        help="Apply mean pooling across the embedding dimension (reduces shape from (N, 896, 3072) to (N, 896)).",
    )

    args = parser.parse_args()

    # Convert pad_value string to appropriate type
    if args.pad_value == "-1":
        pad_value = -1
    else:
        pad_value = args.pad_value

    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process FASTA file and save embeddings
    try:
        print(f"Reading sequences from: {args.input_file}")
        print("Creating Enformer model...")
        model = create_enformer_model()
        print("Retrieving embeddings...")

        embeddings, _, sequence_ids = retrieve_embeddings_from_fasta(
            fasta_path=args.input_file,
            model=model,
            center_sequences=not args.no_center,
            window_size=args.window_size,
            pad_value=pad_value,
            save_path=args.output_file,
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

# python -m retrieve_embeddings.cli     --input-file test.fasta     --output-file embeddings_mean_pool.npz     --mean-pool