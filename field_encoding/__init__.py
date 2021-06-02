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


def decode_ids(ids, width=None, height=None, most_significant_rows=True, shape=None):
    if width is not None and height is not None:
        return decode_ids_wh(ids, width, height, most_significant_rows)
    elif shape is not None:
        return decode_ids_s(ids, shape, most_significant_rows)

    raise Exception("Either provide width and height or shape")


def decode_ids_wh(ids, width, height, most_significant_rows):
    size = width if most_significant_rows else height
    bits = n_bits(size)
    msb, lsb = decode_with_field(ids, bits)
    rows, cols = (msb, lsb) if most_significant_rows else (lsb, msb)

    return rows, cols


def decode_ids_s(ids, shape, most_significant_rows):
    height, width = shape
    return decode_ids_wh(ids, width, height, most_significant_rows)


def encode_coords(rows, cols, width=None, height=None, most_significant_rows=True, shape=None):
    if width is not None and height is not None:
        return encode_coords_wh(rows, cols, width, height, most_significant_rows)
    elif shape is not None:
        return encode_coords_s(rows, cols, shape, most_significant_rows)

    raise Exception("Either provide width and height or shape")


def encode_coords_wh(rows, cols, width, height, most_significant_rows):
    size = width if most_significant_rows else height
    bits = n_bits(size)
    msb, lsb = (rows, cols) if most_significant_rows else (cols, rows)
    ids = encode_with_field(msb, lsb, bits)

    return ids


def encode_coords_s(rows, cols, shape, most_significant_rows):
    height, width = shape
    return encode_coords_wh(rows, cols, width, height, most_significant_rows)


def power_of_2_size(width=None, height=None, most_significant_rows=True, shape=None):
    if width is not None and height is not None:
        return power_of_2_size_wh(width, height, most_significant_rows)
    elif shape is not None:
        return power_of_2_size_s(shape, most_significant_rows)

    raise Exception("Either provide width and height or shape")


def power_of_2_size_wh(width, height, most_significant_rows):
    msb_size = width if most_significant_rows else height
    lsb_size = height if most_significant_rows else width
    msb_bits = n_bits(msb_size)
    lsb_bits = n_bits(lsb_size)

    return int(2**(msb_bits + lsb_bits))


def power_of_2_size_s(shape, most_significant_rows):
    height, width = shape
    return power_of_2_size_wh(width, height, most_significant_rows)


def max_coord_size(width=None, height=None, most_significant_rows=True, shape=None):
    if width is not None and height is not None:
        return max_coord_size_wh(width, height, most_significant_rows)
    elif shape is not None:
        return max_coord_size_s(shape, most_significant_rows)

    raise Exception("Either provide width and height or shape")


def max_coord_size_wh(width, height, most_significant_rows):
    r = height - 1
    c = width - 1
    return encode_coords(r, c, width, height, most_significant_rows) + 1


def max_coord_size_s(shape, most_significant_rows):
    height, width = shape
    return max_coord_size_wh(width, height, most_significant_rows)


def get_augmented_shape(shape):
    bits = n_bits(shape)
    return np.uint32(np.power(2, bits))


def get_encoded_ids(shape, most_significant_rows):
    n_in_rows = shape[0]
    n_in_cols = shape[1]
    rows = np.repeat(np.arange(n_in_rows), n_in_cols)
    cols = np.tile(np.arange(n_in_cols), n_in_rows)
    ids = encode_coords(rows, cols, n_in_rows, n_in_cols, most_significant_rows)
    return ids


