import os
import shutil
import tempfile

import pytest

from ..batch.batch_corrupt import batch_corrupt_image
from ..batch.batch_synth import batch_generate_img_with_filaments


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


@pytest.fixture(params=[2, 10, 20])
def n_img(request):
    return request.param


@pytest.fixture(params=[10, (1, 20)])
def n_filaments(request):
    return request.param


@pytest.fixture(params=[1, 20])
def n_jobs(request):
    return request.param


def test_batch_synth(temp_dir, n_img, n_jobs, corruption_steps, n_filaments):
    batch_generate_img_with_filaments(n_img,
                                      n_filaments=n_filaments,
                                      dir_out=os.path.join(temp_dir, 'input'),
                                      imgshape=(20, 20, 20), n_jobs=n_jobs)
    assert len(os.listdir(os.path.join(temp_dir, 'input'))) == n_img
    batch_corrupt_image(os.path.join(temp_dir, 'input'),
                        os.path.join(temp_dir, 'output'),
                        corruption_steps, n_jobs=n_jobs)
    assert len(os.listdir(os.path.join(temp_dir, 'output'))) == n_img
