import numpy as np
from typing import Any, Dict, Optional, Union, Sequence

class GDict(dict):
    """Utility class to handle dictionaries of arrays as objects."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'GDict' object has no attribute '{name}'")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        res = cls()
        for k, v in d.items():
            if isinstance(v, dict):
                res[k] = cls.from_dict(v)
            else:
                res[k] = v
        return res

class DictArray:
    """A wrapper for a dictionary of arrays, providing array-like indexing."""
    def __init__(self, data: Union[Dict[str, Any], GDict]):
        self.data = data if isinstance(data, GDict) else GDict.from_dict(data)

    def __getitem__(self, index):
        return GDict({k: v[index] for k, v in self.data.items()})

    def __len__(self):
        # Asume que todos los arrays tienen la misma longitud
        for v in self.data.values():
            if hasattr(v, '__len__'):
                return len(v)
        return 0

    @property
    def shape(self):
        for v in self.data.values():
            if hasattr(v, 'shape'):
                return v.shape
        return ()