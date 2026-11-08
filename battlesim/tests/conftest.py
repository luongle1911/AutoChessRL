import numpy as np
import pytest

from battlesim import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def np_rng():
    return np.random.default_rng(12345)
