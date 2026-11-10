import random

import numpy as np
import torch
import yaml


def load_config(path, overrides=()):
    """Read a YAML config and apply `key.sub=value` overrides from the command line."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parse_value(value)
    return cfg


def parse_value(text):
    """YAML value, but '1e-4' is a number (YAML 1.1 would read it as a string)."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
