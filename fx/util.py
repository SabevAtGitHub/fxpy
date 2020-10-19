from contextlib import contextmanager
from time import time


@contextmanager
def timethis(msg='Elapsed'):
    start = time()
    yield
    elapsed = time() - start
    print(f'{msg}: {elapsed:.2f}sec.')