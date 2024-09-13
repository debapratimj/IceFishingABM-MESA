import numpy as np
import pytest
from ice_fishing_abm_1.ice_fishing_abm_random_walker.utils import x_y_to_i_j

def test_x_y_to_i_j():
    assert x_y_to_i_j(1, 2) == (2, 1)
    assert x_y_to_i_j(0, 0) == (0, 0)
    assert x_y_to_i_j(5, 10) == (10, 5)
    assert x_y_to_i_j(-1, -2) == (-2, -1)


if __name__ == "__main__":
    pytest.main()