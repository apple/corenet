#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import unittest

from corenet.utils.check import check


class TestCheck(unittest.TestCase):

    def test_ok(self):
        check(True)
        check(1)
        check([1])

    def test_fail(self):
        with self.assertRaises(AssertionError):
            check(False)
        with self.assertRaises(AssertionError):
            check(0)
        with self.assertRaises(AssertionError):
            check([])

    def test_custom_raise(self):
        with self.assertRaisesRegex(AssertionError, "phooey"):
            check(False, "phooey")
        with self.assertRaisesRegex(ValueError, "phooey"):
            check(False, ValueError("phooey"))
        with self.assertRaisesRegex(AssertionError, "phooey"):
            check(False, lambda: "phooey")
        with self.assertRaisesRegex(ValueError, "phooey"):
            check(False, lambda: ValueError("phooey"))
        with self.assertRaisesRegex(AssertionError, "phooey: 0"):
            check(0, lambda x: f"phooey: {x}")
        with self.assertRaisesRegex(ValueError, "phooey: 0"):
            check(0, lambda x: ValueError(f"phooey: {x}"))
