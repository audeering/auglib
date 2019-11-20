from typing import Any, Union, Callable
import importlib
import os

import oyaml as yaml


class Object:
    r"""Base class for all other classes.

    Derived classes can be turned into a dictionary and dumped to / read
    from a YAML file.

    .. note:: Order of the variables is preserved in the yaml structure.
        Private variables declared starting with a '_' are excluded.

    Example:

        >>> class Foo(Object):
        >>>    def __init__(self, foo: str, bar: int):
        >>>        self.foo = foo
        >>>        self.bar = bar
        >>>        self._private = 'hidden'
        >>> Foo('foo', 1234)
        Foo:
            foo: foo
            bar: 1234

    """
    @staticmethod
    def _to_dict(value: Any):
        if value is None:
            return None
        elif type(value) in [str, int, float, bool]:
            return value
        if isinstance(value, Object):
            return value.to_dict()
        elif isinstance(value, list):
            return [Object._to_dict(item) for item in value]
        elif isinstance(value, tuple):
            return [Object._to_dict(item) for item in value]
        elif isinstance(value, dict):
            return {key: Object._to_dict(val) for key, val in value.items()}
        else:
            return repr(value)

    def to_dict(self) -> dict:
        name = '{}.{}'.format(self.__class__.__module__,
                              self.__class__.__name__)
        return {name: {
            key: Object._to_dict(value) for key, value in
            self.__dict__.items() if not key.startswith('_')
        }}

    @staticmethod
    def _split_key(key: str) -> [str, str]:
        tokens = key.split('.')
        module_name = '.'.join(tokens[:-1])
        class_name = tokens[-1]
        return module_name, class_name

    @staticmethod
    def _get_class(key: str):
        module_name, class_name = Object._split_key(key)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @staticmethod
    def _from_dict(value: Any) -> Any:
        if isinstance(value, list):
            return [Object._from_dict(v) for v in value]
        elif isinstance(value, tuple):
            return tuple([Object._from_dict(v) for v in value])
        elif isinstance(value, dict):
            name = next(iter(value))
            if isinstance(name, str) and name.startswith('auglib.'):
                return Object.from_dict(value)
            else:
                return {k: Object._from_dict(v) for k, v in value.items()}
        else:
            return value

    @staticmethod
    def from_dict(d: dict) -> 'Object':
        name = next(iter(d))
        cls = Object._get_class(name)
        params = {}
        for key, value in d[name].items():
            params[key] = Object._from_dict(value)
        return cls(**params)

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return yaml.dump(self.to_dict())

    def __eq__(self, other):
        return repr(self) == repr(other)

    def dump(self, path: str = None):
        if path is None:
            return yaml.dump(self.to_dict())
        else:
            if not path.endswith('.yaml'):
                path += '.yaml'
            root = os.path.dirname(path)
            if root and not os.path.exists(root):
                os.makedirs(root)
            with open(path, 'w') as fp:
                return yaml.dump(self.to_dict(), fp)

    @staticmethod
    def load(path: str) -> Union['Object', Callable]:
        if not path.endswith('.yaml'):
            path += '.yaml'
        with open(path, 'r') as fp:
            return Object.from_dict(yaml.load(fp, yaml.Loader))
