#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from corenet.utils.registry import Registry


def test_functional_registry() -> None:
    reg = Registry("registry_name")
    reg.register("awesome_dict")(dict)

    assert "awesome_dict" in reg
    assert "awesome_dict(name=hello)" in reg

    obj = reg["awesome_dict(name=hello, type=fifo)"]()

    assert obj == {"name": "hello", "type": "fifo"}


def test_basic_registration() -> None:
    my_registry = Registry("registry_name")

    @my_registry.register("awesome_class_or_func")
    def my_awesome_class_or_func(param):
        pass

    assert "awesome_class_or_func" in my_registry
    assert "awesome_class_or_func(param=value)" in my_registry
