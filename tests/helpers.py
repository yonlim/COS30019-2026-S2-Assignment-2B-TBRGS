"""Shared set-up for the tests: the real dataset is loaded once and reused."""

import functools
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from tbrgs.config import load_config  # noqa: E402


@functools.lru_cache(maxsize=None)
def config():
    return load_config()


@functools.lru_cache(maxsize=None)
def prepared():
    from tbrgs.training import prepare
    return prepare(config())


@functools.lru_cache(maxsize=None)
def system():
    from tbrgs.system import TBRGS
    return TBRGS(config())


def trained(name):
    from tbrgs.models import trained_models
    return name in trained_models(config().path_of("paths", "model_dir"))
