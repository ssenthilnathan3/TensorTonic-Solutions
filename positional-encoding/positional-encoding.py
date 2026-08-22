import math
import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pe = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range((d_model + 1) // 2):
            w = 1 / (base ** ((2 * i) / d_model))
            angle = pos * w

            sin_idx = 2 * i
            cos_idx = 2 * i + 1

            if sin_idx < d_model:
                pe[pos, sin_idx] = math.sin(angle)

            if cos_idx < d_model:
                pe[pos, cos_idx] = math.cos(angle)

    return pe