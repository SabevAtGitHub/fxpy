import pytest

import os
import sys 

import numpy as np
import pandas as pd

import fx

def test_me():
    assert True

def test_update_on():
    first = pd.DataFrame()
    second = pd.DataFrame()

    result = fx.update_on(first, second, on='A')

    assert type(result) == type(second)
    # assert type re