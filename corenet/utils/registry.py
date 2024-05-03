#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from corenet.utils import logger
from corenet.utils.import_utils import import_modules_from_folder

RegistryItem = TypeVar("RegistryItem", bound=Callable)


class Registry:
    """
    A key/object registry class.
    This class is used to do Dependency Injection in configs,
    so when you write "resnet" in a config, it knows which module to load.
    You can potentially provide a `base_class` to ensures that all items
    in the registry are of type `base_class`.

    Registry also allows for passing arguments to a registered item:
    For example: "top1" -> "top1(pred=logits)"

    Usage:
    >>> my_registry = Registry("registry_name")
    >>> @my_registry.register("awesome_class_or_func")
    ... def my_awesome_class_or_func():
    ...    pass
    >>> assert "awesome_class_or_func" in my_registry

    It allows for vanilla key/object definition as well as functional argument injection:
    >>> reg = Registry("registry_name")
    >>> reg.register("awesome_dict")(dict)
    >>> reg["awesome_dict(name=hello, type=fifo)]()
    {'name': 'hello', 'type': 'fifo'}
    """

    def __init__(
        self,
        registry_name: str,
        base_class: Optional[type] = None,
        separator: Optional[str] = ":",
        lazy_load_dirs: Optional[List[str]] = None,
        internal_dirs: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Args:
            registry_name: registry name, used for debugging and error messages
            base_class: If provided, will ensure that all items inside the registry
                are of type `base_class`.
            separator: Separator between name and type in `register` function.
            lazy_load_dirs: If provided, will load all directories under these
                directories when inspecting for the modules of the registry.
        """
        self.registry_name = registry_name
        self.base_class = base_class
        self.registry = {}
        # For debugging purposes we want to throw a warning if someone accesses
        # arguments before registering all items.
        self.arguments_accessed = False
        self.separator = separator
        # Lazy loading to get rid of possible circular dependencies
        self._modules_loaded = False
        self._lazy_load_dirs = lazy_load_dirs
        self.internal_dirs = internal_dirs
        if self._lazy_load_dirs is None:
            self._lazy_load_dirs = []

    def _load_all(self) -> None:
        """
        This function allows for lazily loading modules from pre-specified directories.
        The main reason for its existence is to prevent circular imports.

        This function should be called before any "pull/get" kind of action from
        Registry to make sure it has loaded all registered models, which is pretty
        much any operation except for "register".

        If self._modules_loaded is not True, it will load all modules under
        self._lazy_load_dirs.
        """
        if not self._modules_loaded:
            self._modules_loaded = True
            for dir_name in sorted(self._lazy_load_dirs):
                import_modules_from_folder(dir_name, extra_roots=self.internal_dirs)

    def items(self) -> List[Tuple[str, RegistryItem]]:
        self._load_all()
        return self.registry.items()

    def keys(self) -> List[str]:
        self._load_all()
        return self.registry.keys()

    def __iter__(self) -> Iterable[str]:
        self._load_all()
        return iter(self.registry)

    def __getitem__(self, key: Union[Tuple[str, str], str]) -> RegistryItem:
        self._load_all()

        type_ = None
        if isinstance(key, Tuple) and len(key) == 2:
            key, type_ = key

        assert isinstance(
            key, str
        ), f"Key should be an instance of string. Got {type(key)}"
        name, params = self.parse_key(key)
        if type_:
            name = f"{type_}{self.separator}{name}"

        if name not in self.registry:
            registry_keys = list(self.registry.keys())
            temp_str = (
                f"\n{name} not yet supported in {self.registry_name} registry."
                f"\nSupported values are:"
            )
            for i, supp_val in enumerate(registry_keys):
                temp_str += f"\n\t {i}: {supp_val}"
            logger.error(temp_str + "\n")

        reg_item = self.registry[name]

        if params:
            reg_item = partial(reg_item, **params)
        return reg_item

    def __contains__(self, key: str) -> bool:
        self._load_all()
        name, _ = self.parse_key(key)
        return name in self.registry

    def register(self, name: str, type: str = "") -> Callable:
        if type:
            name = "{}{}{}".format(type, self.separator, name)

        if self.arguments_accessed:
            # TODO: do we really want an error here?
            logger.error(
                f"Found item `{name}` being registered after all_item_arguments"
                f" was called for `{self.registry_name}` registry."
            )

        def register_with_name(item: RegistryItem) -> RegistryItem:
            if name in self.registry:
                raise ValueError(
                    "Cannot register duplicate {} ({})".format(self.registry_name, name)
                )
            if self.base_class and not issubclass(item, self.base_class):
                raise ValueError(
                    "{} class ({}: {}) must extend {}".format(
                        self.registry_name, name, item.__name__, self.base_class
                    )
                )

            self.registry[name] = item
            return item

        return register_with_name

    def all_arguments(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Iterates through all items and fetches their arguments.

        Note: make sure that all items are already registered before calling this method.
        """
        self._load_all()
        self.arguments_accessed = True

        for _, item in self.items():
            parser = item.add_arguments(parser)

        return parser

    def parse_key(self, key: str) -> Tuple[str, Dict[str, str]]:
        """
        Parses `key` which can contain arguments in the form of:
        <key_name>(arg1=value1, arg2=value2, ...)

        Returns:
            Tuple: (base_name: str, parameters: dict)
        """
        name = key.split("(")[0]

        params = {}
        if "(" in key:
            params_str = key.split("(")[1].split(")")[0]

            try:
                params = dict(
                    [
                        [x.strip() for x in arg.split("=")]
                        for arg in params_str.split(",")
                    ]
                )
            except Exception as e:
                logger.error(
                    "Could not correctly parse key parameters `{}` for registry {}."
                    " Please make sure to key parameters have the format:"
                    " <key_name>(arg1=value1, arg2=value2, ...)".format(
                        key, self.registry_name
                    )
                )
                raise e

        return name, params
