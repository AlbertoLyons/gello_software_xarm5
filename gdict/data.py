# Importación de liberías necesarias
import numpy as np
from typing import Any, Dict, Union
"""
data.py
Clase GDict y DictArray para manejar diccionarios de arrays como objetos.
"""
class GDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    # Función para acceder a los elementos del diccionario como atributos
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'GDict' object has no attribute '{name}'")
    # Función para crear un GDict a partir de un diccionario anidado
    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        res = cls()
        for k, v in d.items():
            if isinstance(v, dict):
                res[k] = cls.from_dict(v)
            else:
                res[k] = v
        return res
"""
Clase DictArray para manejar diccionarios de arrays como objetos con indexación.
"""
class DictArray:
    def __init__(self, data: Union[Dict[str, Any], GDict]):
        self.data = data if isinstance(data, GDict) else GDict.from_dict(data)
    # Función que obtiene un elemento del diccionario de arrays utilizando indexación
    def __getitem__(self, index):
        return GDict({k: v[index] for k, v in self.data.items()})
    # Función que devuelve la longitud de los arrays en el diccionario
    def __len__(self):
        # Asume que todos los arrays tienen la misma longitud
        for v in self.data.values():
            if hasattr(v, '__len__'):
                return len(v)
        return 0
    # Función que devuelve la forma de los arrays en el diccionario
    @property
    def shape(self):
        for v in self.data.values():
            if hasattr(v, 'shape'):
                return v.shape
        return ()