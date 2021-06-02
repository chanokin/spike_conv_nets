import numpy as np


def n_bits(max_val):
    return np.uint32(np.ceil(np.log2(max_val)))


def generate_mask(shift):
    return (1 << shift) - 1


def decode_with_field(ids, shift):
    mask = generate_mask(shift)
    return (np.right_shift(ids, shift),
            np.bitwise_and(ids, mask))


def encode_with_field(msf, lsf, shift):
    mask = generate_mask(shift)
    return np.bitwise_or(np.left_shift(msf, shift),
                         np.bitwise_and(lsf, mask))


def decode_ids(ids, width, height, most_significant_rows):
    size = width if most_significant_rows else height
    bits = n_bits(size)
    msb, lsb = decode_with_field(ids, bits)
    rows, cols = (msb, lsb) if most_significant_rows else (lsb, msb)

    return rows, cols


def encode_coords(rows, cols, width, height, most_significant_rows):
    size = width if most_significant_rows else height
    bits = n_bits(size)
    msb, lsb = (rows, cols) if most_significant_rows else (cols, rows)
    ids = encode_with_field(msb, lsb, bits)

    return ids


def power_of_2_size(width, height, most_significant_rows):
    msb_size = width if most_significant_rows else height
    lsb_size = height if most_significant_rows else width
    msb_bits = n_bits(msb_size)
    lsb_bits = n_bits(lsb_size)

    return int(2**(msb_bits + lsb_bits))


def get_augmented_shape(shape):
    bits = n_bits(shape)
    return np.uint32(np.power(2, bits))



