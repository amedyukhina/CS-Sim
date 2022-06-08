import os
import shutil
import tempfile

import pytest

from ..batch.batch_synth import batch_generate_img_with_lines


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


@pytest.fixture(params=[2, 10, 20])
def n_img(request):
    return request.param


@pytest.fixture(params=[1, 20])
def n_jobs(request):
    return request.param


def test_batch_synth(temp_dir, n_img, n_jobs):
    batch_generate_img_with_lines(n_img, dir_out=temp_dir, imgshape=(20, 20, 20), n_jobs=n_jobs)
    assert len(os.listdir(temp_dir)) == n_img
