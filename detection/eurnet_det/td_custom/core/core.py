"""
Mainly copy from https://github.com/DeepGraphLearning/torchdrug
"""
import re
import types
import inspect
from collections import defaultdict
from contextlib import contextmanager

from decorator import decorator

class _MetaContainer(object):
    """
    Meta container that maintains meta types about members.
    The meta type of each member is tracked when a member is assigned.
    We use a context manager to define the meta types for a bunch of assignment.
    The meta types are stored as a dict in ``instance.meta_dict``,
    where keys are member names and values are meta types.
    >>> class MyClass(_MetaContainer):
    >>>     ...
    >>> instance = MyClass()
    >>> with instance.context("important"):
    >>>     instance.value = 1
    >>> assert instance.meta_dict["value"] == "important"
    Members assigned with :meth:`context(None) <context>` or without a context won't be tracked.
    >>> instance.random = 0
    >>> assert "random" not in instance.meta_dict
    You can also restrict available meta types by defining a set :attr:`_meta_types` in the derived class.
    .. note::
        Meta container also supports auto inference of meta types.
        This can be enabled by setting :attr:`enable_auto_context` to ``True`` in the derived class.
        Once auto inference is on, any member without an explicit context will be recognized through their name prefix.
        For example, ``instance.node_value`` will be recognized as ``node`` if ``node`` is defined in ``meta_types``.
        This may make code hard to maintain. Use with caution.
    """

    _meta_types = set()
    enable_auto_context = False

    def __init__(self, meta_dict=None, **kwargs):
        if meta_dict is None:
            meta_dict = {}
        else:
            meta_dict = meta_dict.copy()

        self._setattr("_meta_contexts", set())
        self._setattr("meta_dict", meta_dict)
        for k, v in kwargs.items():
            self._setattr(k, v)

    @contextmanager
    def context(self, type):
        """
        Context manager for assigning members with a specific meta type.
        """
        if type is not None and self._meta_types and type not in self._meta_types:
            raise ValueError("Expect context type in %s, but got `%s`" % (self._meta_types, type))
        self._meta_contexts.add(type)
        yield
        self._meta_contexts.remove(type)

    def __setattr__(self, key, value):
        if hasattr(self, "meta_dict"):
            types = self._meta_contexts
            if not types and self.enable_auto_context:
                for type in self._meta_types:
                    if key.startswith(type):
                        types.append(type)
                if len(types) > 1:
                    raise ValueError("Auto context found multiple contexts for key `%s`. "
                                     "If this is desired, set `enable_auto_context` to False "
                                     "and manually specify the context. " % key)
            if types:
                self.meta_dict[key] = types.copy()
        self._setattr(key, value)

    def __delattr__(self, key):
        if hasattr(self, "meta_dict") and key in self.meta_dict:
            del self.meta_dict[key]
            del self.data_dict[key]
        super(_MetaContainer, self).__delattr__(self, key)

    def _setattr(self, key, value):
        return super(_MetaContainer, self).__setattr__(key, value)

    @property
    def data_dict(self):
        """A dict that maps tracked names to members."""
        return {k: getattr(self, k) for k in self.meta_dict}

    def data_by_meta(self, include=None, exclude=None):
        """
        Return members based on the specific meta types.
        Parameters:
            include (list of string, optional): meta types to include
            exclude (list of string, optional): meta types to exclude
        Returns:
            (dict, dict): data member dict and meta type dict
        """
        if include is None and exclude is None:
            return self.data_dict, self.meta_dict

        include = self._standarize_type(include)
        exclude = self._standarize_type(exclude)
        types = include or set().union(*self.meta_dict.values())
        types = types - exclude
        data_dict = {}
        meta_dict = {}
        for k, v in self.meta_dict.items():
            if v.issubset(types):
                data_dict[k] = getattr(self, k)
                meta_dict[k] = v
        return data_dict, meta_dict

    def _standarize_type(self, types):
        if types is None:
            types = set()
        elif isinstance(types, str):
            types = {types}
        else:
            types = set(types)
        return types