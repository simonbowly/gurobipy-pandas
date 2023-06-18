import unittest

import gurobipy as gp
import numpy as np
from pandas.api.types import is_extension_array_dtype, pandas_dtype

from gurobipy_pandas.extension import (
    GurobiLinExprDtype,
    GurobiMObjectArray,
    GurobiVarDtype,
)

from .utils import GurobiModelTestCase


class TestRegisteredGurobiDtypes(unittest.TestCase):
    def test_gpvar(self):
        dtype = pandas_dtype("gpvar")
        self.assertTrue(is_extension_array_dtype(dtype))

    def test_gplinexpr(self):
        dtype = pandas_dtype("gplinexpr")
        self.assertTrue(is_extension_array_dtype(dtype))


class TestCreateGurobiMObjectArray(GurobiModelTestCase):
    def test_init(self):
        # Constructor takes ownership of an MVar
        mvar = self.model.addMVar((10,))
        arr = GurobiMObjectArray(mvar)
        self.assertEqual(len(arr), 10)

    def test_dtype(self):
        mvar = self.model.addMVar((10,))
        arr = GurobiMObjectArray(mvar)
        self.assertIsInstance(arr.dtype, GurobiVarDtype)


class TestGurobiMObjectArrayGetItem(GurobiModelTestCase):
    def setUp(self):
        super().setUp()
        mvar = self.model.addMVar((10,), name="x")
        self.arr = GurobiMObjectArray(mvar)
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
            self.assertIsInstance(obj, GurobiMObjectArray)
            self.assertEqual(len(obj), 1)
            self.assertEqual(obj[0].VarName, f"x[{i}]")

    def test_slice_2(self):
        # For slice ``key``, return an instance of ``ExtensionArray``, even
        # if the slice is length 0 or 1.
        obj = self.arr[1:6:2]
        self.assertIsInstance(obj, GurobiMObjectArray)
        self.assertEqual(len(obj), 3)
        for i in range(3):
            self.assertEqual(obj[i].VarName, f"x[{2*i + 1}]")

    def test_mask(self):
        # For a boolean mask, return an instance of ``ExtensionArray``, filtered
        # to the values where ``item`` is True.
        mask = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).astype(bool)
        obj = self.arr[mask]
        self.assertIsInstance(obj, GurobiMObjectArray)
        self.assertEqual(len(obj), 5)
        for i in range(5):
            self.assertEqual(obj[i].VarName, f"x[{2*i + 1}]")


class TestGurobiMObjectArrayTake(GurobiModelTestCase):
    def setUp(self):
        super().setUp()
        mvar = self.model.addMVar((8,), name="x")
        self.arr = GurobiMObjectArray(mvar)
        self.model.update()

    def test_1(self):
        positions = [1, 3, 4, 7]
        obj = self.arr.take(np.array(positions))
        self.assertIsInstance(obj, GurobiMObjectArray)
        self.assertEqual(len(obj), 4)
        for i in range(4):
            self.assertEqual(obj[i].VarName, f"x[{positions[i]}]")


class TestGurobiMObjectArrayCopy(GurobiModelTestCase):
    def test_1(self):
        mvar = self.model.addMVar((20,))
        arr = GurobiMObjectArray(mvar)
        arr_copy = arr.copy()
        self.assertIsNot(arr, arr_copy)
        for i in range(20):
            self.assertTrue(arr[i].sameAs(arr_copy[i]))

    # TODO add more tests here after __iadd__ works


# Only minimal tests for arithmetic operators here. The array type just
# delegates to gurobi M* class operations, and more extensive testing is done on
# the resulting Series in test_operators.


class TestGurobiMObjectArrayAdd(GurobiModelTestCase):
    def test_vararray_plus_scalar(self):
        vararr = GurobiMObjectArray(self.model.addMVar((5,)))
        self.assertIsInstance(vararr.dtype, GurobiVarDtype)
        learr = vararr + 2.0
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(learr[i], vararr[i] + 2.0)

    def test_vararray(self):
        x = GurobiMObjectArray(self.model.addMVar((5,)))
        y = GurobiMObjectArray(self.model.addMVar((5,)))
        learr = x + y
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(learr[i], x[i] + y[i])


class TestGurobiMObjectArrayRadd(GurobiModelTestCase):
    def test_var_plus_vararray(self):
        vararr = GurobiMObjectArray(self.model.addMVar((5,)))
        x = self.model.addVar()
        self.assertIsInstance(vararr.dtype, GurobiVarDtype)
        learr = x + vararr
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            # Term ordering is different as radd just delegates to add
            self.assert_linexpr_equal(learr[i], vararr[i] + x)


class TestGurobiMObjectArrayIadd(GurobiModelTestCase):
    def test_scalar(self):
        mvar = self.model.addMVar((5,))
        arr = GurobiMObjectArray(mvar)
        self.assertIsInstance(arr.dtype, GurobiVarDtype)
        arr += 5.0
        self.assertIsInstance(arr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(arr[i], mvar[i].item() + 5.0)

    def test_vararray(self):
        x = self.model.addMVar((5,))
        y = self.model.addMVar((5,))
        xarr = GurobiMObjectArray(x)
        yarr = GurobiMObjectArray(y)
        xarr += yarr
        self.assertIsInstance(xarr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(xarr[i], x[i].item() + y[i].item())


class TestGurobiMObjectArraySub(GurobiModelTestCase):
    def test_vararray_minus_linexpr(self):
        vararr = GurobiMObjectArray(self.model.addMVar((5,)))
        x = self.model.addVar()
        self.assertIsInstance(vararr.dtype, GurobiVarDtype)
        learr = vararr - (x + 1.0)
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(learr[i], vararr[i] - x - 1.0)

    def test_vararray(self):
        x = GurobiMObjectArray(self.model.addMVar((5,)))
        y = GurobiMObjectArray(self.model.addMVar((5,)))
        learr = x - y
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(learr[i], x[i] - y[i])


class TestGurobiMObjectArrayRsub(GurobiModelTestCase):
    def test_linexpr_minus_vararray(self):
        vararr = GurobiMObjectArray(self.model.addMVar((5,)))
        x = self.model.addVar()
        self.assertIsInstance(vararr.dtype, GurobiVarDtype)
        learr = (2 * x + 4) - vararr
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            # Term ordering is different as radd just delegates to add
            self.assert_linexpr_equal(learr[i], -vararr[i] + 2 * x + 4)


class TestGurobiMObjectArrayIsub(GurobiModelTestCase):
    def test_mvar_0d(self):
        mvar = self.model.addMVar((5,))
        arr = GurobiMObjectArray(mvar)
        self.assertIsInstance(arr.dtype, GurobiVarDtype)
        arr -= mvar[2]
        self.assertIsInstance(arr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(arr[i], mvar[i].item() - mvar[2].item())

    def test_vararray(self):
        x = self.model.addMVar((5,))
        y = self.model.addMVar((5,))
        xarr = GurobiMObjectArray(x)
        yarr = GurobiMObjectArray(y)
        xarr -= yarr
        self.assertIsInstance(xarr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(xarr[i], x[i].item() - y[i].item())
