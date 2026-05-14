import yaml
from pathlib import Path

class Config:
    def __init__(self, path="configs/default.yaml"):
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f) or {}

    def get(self, key, default=None):
        keys = key.split(".")
        value = self.data
        for k in keys:
            if not isinstance(value, dict):
                return default
            if k not in value:
                return default
            value = value[k]
        return value

    def set(self, key, value):
        keys = key.split(".")
        node = self.data
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.data, f, sort_keys=False, allow_unicode=False)
