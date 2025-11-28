"""
Tests for sequence tensor centering and padding functionality.
"""

import pytest
import torch
from retrieve_embeddings.util import (
    center_sequence_tensor_in_window,
    validate_sequence,
    dna_string_to_indices,
)


class TestCenterSequenceTensorInWindow:
    """Test suite for center_sequence_tensor_in_window function."""

    def test_padding_with_default_value(self) -> None:
        """Test padding with default -1 value."""
        seq = torch.tensor([0, 1, 2, 3])  # ACGT
        window_size = 10
        result = center_sequence_tensor_in_window(seq, window_size=window_size)

        assert result.shape[0] == window_size
        assert torch.all(result[:3] == -1)  # Left padding
        assert torch.all(result[3:7] == torch.tensor([0, 1, 2, 3]))  # Original sequence
        assert torch.all(result[7:] == -1)  # Right padding

    def test_padding_with_custom_value(self) -> None:
        """Test padding with custom pad_value (e.g., 4 for N)."""
        seq = torch.tensor([0, 1, 2])  # ACG
        window_size = 8
        pad_value = 4
        result = center_sequence_tensor_in_window(
            seq, window_size=window_size, pad_value=pad_value
        )

        assert result.shape[0] == window_size
        assert torch.all(result[:2] == pad_value)  # Left padding
        assert torch.all(result[2:5] == torch.tensor([0, 1, 2]))  # Original sequence
        assert torch.all(result[5:] == pad_value)  # Right padding

    def test_centering_uneven_padding(self) -> None:
        """Test that when padding is uneven, extra padding goes to the right."""
        seq = torch.tensor([0, 1])  # AC
        window_size = 5
        result = center_sequence_tensor_in_window(seq, window_size=window_size, pad_value=-1)

        assert result.shape[0] == window_size
        # Should be: [-1, 0, 1, -1, -1] (left=1, right=2)
        assert result[0] == -1
        assert result[1] == 0
        assert result[2] == 1
        assert result[3] == -1
        assert result[4] == -1

    def test_exact_window_size(self) -> None:
        """Test sequence that is exactly the window size."""
        seq = torch.tensor([0, 1, 2, 3, 4])
        window_size = 5
        result = center_sequence_tensor_in_window(seq, window_size=window_size)

        assert result.shape[0] == window_size
        assert torch.all(result == seq)  # No padding needed

    def test_sequence_longer_than_window_raises_error(self) -> None:
        """Test that ValueError is raised when sequence is longer than window."""
        seq = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        window_size = 5

        with pytest.raises(ValueError, match="exceeds window size"):
            center_sequence_tensor_in_window(seq, window_size=window_size)

    def test_sequence_longer_than_window_uneven_raises_error(self) -> None:
        """Test that ValueError is raised when sequence is longer (uneven case)."""
        seq = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        window_size = 4

        with pytest.raises(ValueError, match="exceeds window size"):
            center_sequence_tensor_in_window(seq, window_size=window_size)

    def test_single_element_sequence(self) -> None:
        """Test padding a single element sequence."""
        seq = torch.tensor([2])  # G
        window_size = 5
        result = center_sequence_tensor_in_window(seq, window_size=window_size, pad_value=-1)

        assert result.shape[0] == window_size
        assert result[2] == 2  # Centered element
        assert torch.all(result[:2] == -1)  # Left padding
        assert torch.all(result[3:] == -1)  # Right padding

    def test_empty_sequence(self) -> None:
        """Test padding an empty sequence."""
        seq = torch.tensor([], dtype=torch.long)
        window_size = 5
        result = center_sequence_tensor_in_window(seq, window_size=window_size, pad_value=-1)

        assert result.shape[0] == window_size
        assert torch.all(result == -1)  # All padding

    def test_large_window_size(self) -> None:
        """Test with Enformer's default window size."""
        seq = torch.randint(0, 5, (1000,))  # Random sequence
        window_size = 196_608
        result = center_sequence_tensor_in_window(seq, window_size=window_size, pad_value=-1)

        assert result.shape[0] == window_size
        # Check that original sequence is centered
        pad_left = (window_size - 1000) // 2
        assert torch.all(result[pad_left : pad_left + 1000] == seq)
        # Check padding
        assert torch.all(result[:pad_left] == -1)
        assert torch.all(result[pad_left + 1000 :] == -1)

    def test_preserves_dtype(self) -> None:
        """Test that the function preserves the input tensor dtype."""
        seq = torch.tensor([0, 1, 2], dtype=torch.long)
        result = center_sequence_tensor_in_window(seq, window_size=5, pad_value=-1)

        assert result.dtype == torch.long

    def test_multiple_pad_values(self) -> None:
        """Test different pad values produce correct results."""
        seq = torch.tensor([0, 1])
        window_size = 6

        for pad_value in [-1, 0, 4, 99]:
            result = center_sequence_tensor_in_window(
                seq, window_size=window_size, pad_value=pad_value
            )
            assert result.shape[0] == window_size
            assert torch.all(result[:2] == pad_value)  # Left padding
            assert torch.all(result[2:4] == torch.tensor([0, 1]))  # Original sequence
            assert torch.all(result[4:] == pad_value)  # Right padding

    def test_centering_symmetry(self) -> None:
        """Test that centering is symmetric when possible."""
        seq = torch.tensor([0, 1, 2, 3])
        window_size = 10
        result = center_sequence_tensor_in_window(seq, window_size=window_size, pad_value=-1)

        # Should have equal padding on both sides: 3 left, 3 right (total 6)
        # Sequence length 4, window 10, so padding = 6, split as 3+3
        pad_left = 3
        seq_len = 4
        pad_right = 3
        assert result.shape[0] == window_size
        assert torch.all(result[:pad_left] == -1)  # Left padding
        assert torch.all(result[pad_left : pad_left + seq_len] == seq)  # Original sequence
        assert torch.all(result[pad_left + seq_len :] == -1)  # Right padding
        assert len(result[pad_left + seq_len :]) == pad_right  # Verify right padding length


class TestValidateSequence:
    """Test suite for validate_sequence function."""

    def test_valid_sequence_no_window(self) -> None:
        """Test validation passes for valid sequence without window size check."""
        validate_sequence("ACGTN-")

    def test_valid_sequence_within_window(self) -> None:
        """Test validation passes for valid sequence within window size."""
        validate_sequence("ACGT", window_size=10)

    def test_valid_sequence_exact_window(self) -> None:
        """Test validation passes for sequence exactly at window size."""
        validate_sequence("ACGTACGT", window_size=8)

    def test_invalid_character_raises_error(self) -> None:
        """Test that ValueError is raised for invalid characters."""
        with pytest.raises(ValueError, match="Invalid character"):
            validate_sequence("ACGTX")

    def test_multiple_invalid_characters(self) -> None:
        """Test that all invalid characters are reported."""
        with pytest.raises(ValueError, match="Invalid character"):
            validate_sequence("ACGTXYZ")
        # Error message should mention X, Y, Z

    def test_sequence_exceeds_window_raises_error(self) -> None:
        """Test that ValueError is raised when sequence exceeds window size."""
        with pytest.raises(ValueError, match="exceeds window size"):
            validate_sequence("ACGTACGT", window_size=5)

    def test_custom_allowed_chars(self) -> None:
        """Test validation with custom allowed characters."""
        # Should pass with custom allowed chars
        validate_sequence("ACGT", allowed_chars="ACGTN-")

        # Should fail with custom allowed chars
        with pytest.raises(ValueError, match="Invalid character"):
            validate_sequence("ACGTN", allowed_chars="ACGT")  # N not allowed

    def test_case_insensitive_validation(self) -> None:
        """Test that validation is case-insensitive."""
        validate_sequence("acgt")  # Lowercase should pass
        validate_sequence("AcGt")  # Mixed case should pass


class TestDnaStringToIndicesValidation:
    """Test suite for dna_string_to_indices with validation."""

    def test_valid_sequence_passes(self) -> None:
        """Test that valid sequences pass validation."""
        result = dna_string_to_indices("ACGT", validate=True, window_size=10)
        expected = torch.tensor([0, 1, 2, 3])
        assert torch.all(result == expected)

    def test_invalid_character_raises_error(self) -> None:
        """Test that invalid characters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid character"):
            dna_string_to_indices("ACGTX", validate=True)

    def test_sequence_exceeds_window_raises_error(self) -> None:
        """Test that sequences exceeding window size raise ValueError."""
        with pytest.raises(ValueError, match="exceeds window size"):
            dna_string_to_indices("ACGTACGT", validate=True, window_size=5)

    def test_validation_can_be_disabled(self) -> None:
        """Test that validation can be disabled."""
        # Should not raise error even with invalid char if validate=False
        result = dna_string_to_indices("ACGTX", validate=False)
        # X will be mapped to 4 (N) by default
        assert result[-1] == 4

