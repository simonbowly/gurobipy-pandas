import unittest

import gurobipy as gp
import numpy as np
from pandas.api.types import is_extension_array_dtype, pandas_dtype

from gurobipy_pandas.extension import GurobiVarArray, GurobiVarDtype

from .utils import GurobiModelTestCase


class TestGurobiVarArray(unittest.TestCase):
    def test_registered_type(self):
        dtype = pandas_dtype("gpvar")
        self.assertTrue(is_extension_array_dtype(dtype))


class TestCreateGurobiVarArray(GurobiModelTestCase):
    def test_init(self):
        # Constructor takes ownership of an MVar
        mvar = self.model.addMVar((10,))
        arr = GurobiVarArray(mvar)
        self.assertEqual(len(arr), 10)

    def test_dtype(self):
        mvar = self.model.addMVar((10,))
        arr = GurobiVarArray(mvar)
        self.assertIsInstance(arr.dtype, GurobiVarDtype)


class TestGurobiVarArrayGetItem(GurobiModelTestCase):
    def setUp(self):
        super().setUp()
        mvar = self.model.addMVar((10,), name="x")
        self.arr = GurobiVarArray(mvar)
        self.model.update()

    def test_scalar(self):
        # For scalar ``item``, return a scalar value suitable for the array's
        # type. This should be an instance of ``self.dtype.type``.
        for i in range(10):
            obj = self.arr[i]
            self.assertIsInstance(obj, gp.Var)
            self.assertEqual(obj.VarName, f"x[{i}]")

    def test_slice_1(self):
        # For slice ``key``, return an instance of ``ExtensionArray``, even
        # if the slice is length 0 or 1.
        for i in range(10):
            obj = self.arr[i : i + 1]
            self.assertIsInstance(obj, GurobiVarArray)
            self.assertEqual(len(obj), 1)
            self.assertEqual(obj[0].VarName, f"x[{i}]")

    def test_slice_2(self):
        # For slice ``key``, return an instance of ``ExtensionArray``, even
        # if the slice is length 0 or 1.
        obj = self.arr[1:6:2]
        self.assertIsInstance(obj, GurobiVarArray)
        self.assertEqual(len(obj), 3)
        for i in range(3):
            self.assertEqual(obj[i].VarName, f"x[{2*i + 1}]")

    def test_mask(self):
        # For a boolean mask, return an instance of ``ExtensionArray``, filtered
        # to the values where ``item`` is True.
        mask = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).astype(bool)
        obj = self.arr[mask]
        self.assertIsInstance(obj, GurobiVarArray)
        self.assertEqual(len(obj), 5)
        for i in range(5):
            self.assertEqual(obj[i].VarName, f"x[{2*i + 1}]")


class TestGurobiVarArrayTake(GurobiModelTestCase):
    def setUp(self):
        super().setUp()
        mvar = self.model.addMVar((8,), name="x")
        self.arr = GurobiVarArray(mvar)
        self.model.update()

    def test_1(self):
        positions = [1, 3, 4, 7]
        obj = self.arr.take(np.array(positions))
        self.assertIsInstance(obj, GurobiVarArray)
        self.assertEqual(len(obj), 4)
        for i in range(4):
            self.assertEqual(obj[i].VarName, f"x[{positions[i]}]")


class TestGurobiVarArrayCopy(GurobiModelTestCase):
    def test_1(self):
        mvar = self.model.addMVar((20,))
        arr = GurobiVarArray(mvar)
        arr_copy = arr.copy()
        self.assertIsNot(arr, arr_copy)
        self.assertIsNot(arr.mvar, arr_copy.mvar)
        for i in range(20):
            self.assertTrue(arr[i].sameAs(arr_copy[i]))


class TestGurobiMObjectArrayAdd(GurobiModelTestCase):
    # Fairly minimal tests here. The Array just delegates to gurobi M* class
    # operations, and more extensive testing is done in test_operators.
    def test_var_plus_scalar(self):
        vararr = GurobiVarArray(self.model.addMVar((5,)))
        learr = vararr + 2.0
        for i in range(5):
            self.assert_linexpr_equal(learr[i], vararr[i] + 2.0)
