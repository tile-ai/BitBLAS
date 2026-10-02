# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import pytest

from bitblas.quantization.utils import interleave_weight


def _interleave_decoder_reference(words, nbits, target_dtype):
    """Place fields in the byte/nibble order consumed by the LOP3 decoders."""
    result = np.zeros_like(words)
    for idx in np.ndindex(words.shape):
        word = int(words[idx])
        output = 0
        for element in range(32 // nbits):
            value = (word >> (element * nbits)) & ((1 << nbits) - 1)
            if target_dtype == "float16":
                # Each eight-element decoder load splits even and odd fields.
                field = (element // 8) * 8 + (element % 2) * 4 + (element % 8) // 2
                shift = field * nbits
            elif nbits == 1:
                # The int8 decoder expands each pair of bytes into four lanes.
                shift = ((element // 16) * 16 + (element % 2) * 8 + ((element % 4) // 2) * 4 +
                         (element % 16) // 4)
            else:
                shift = (element % 4) * 8 + (element // 4) * nbits
            output |= value << shift
        result[idx] = output
    return result.view(np.int8)


@pytest.mark.parametrize("nbits", [1, 2, 4])
@pytest.mark.parametrize("target_dtype", ["float16", "int8"])
@pytest.mark.parametrize("storage_dtype", [np.int8, np.uint8, np.int32, np.uint32])
@pytest.mark.parametrize("basis_vectors", [True, False])
def test_interleave_weight_matches_decoder(nbits, target_dtype, storage_dtype, basis_vectors):
    if basis_vectors:
        # Every input bit, including the sign bit, must reach its own output bit.
        words = (np.uint32(1) << np.arange(32, dtype=np.uint32)).reshape(32, 1)
    else:
        rng = np.random.default_rng(0)
        words = rng.integers(0, 1 << 32, size=(4, 32), dtype=np.uint32)
        words[0, :2] = [0, 0xFFFFFFFF]
    packed = words.view(storage_dtype)
    original = packed.copy()
    expected = _interleave_decoder_reference(words, nbits, target_dtype)

    actual = interleave_weight(packed, nbits, target_dtype)

    assert actual.dtype == np.int8
    assert actual.nbytes == packed.nbytes
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(packed, original)
