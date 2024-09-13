import numpy as np

def x_y_to_i_j(x: int, y: int) -> tuple[int, int]:
    """
    Mesa coordinate system counts axis differently than numpy.
    """
    i, j = y, x

    return i, j