import yaml
from pathlib import Path

class Config:
    def __init__(self, path="configs/default.yaml"):
        with open(path, "r") as f:
            self.data = yaml.safe_load(f)

    def get(self, key, default=None):
        keys = key.split(".")
        value = self.data
        for k in keys:
            value = value.get(k, {})
        return value or default
