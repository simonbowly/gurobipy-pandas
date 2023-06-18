import unittest

import gurobipy as gp
import numpy as np
from pandas.api.types import is_extension_array_dtype, pandas_dtype

from gurobipy_pandas.extension import (
    GurobiLinExprDtype,
    GurobiMObjectArray,
    GurobiQuadExprDtype,
    GurobiVarDtype,
)

from .utils import GurobiModelTestCase


def create_gurobimobjectarray(mobj, nan_mask=None):
    # Helper function: creates a GurobiMObjectArray. If nan_mask is not
    # provided, create one such that the returned object has no missing values.
    if nan_mask is None:
        nan_mask = np.zeros(mobj.shape, dtype=bool)
    return GurobiMObjectArray(mobj, nan_mask)


class TestRegisteredGurobiDtypes(unittest.TestCase):
    def test_gpvar(self):
        dtype = pandas_dtype("gpvar")
        self.assertTrue(is_extension_array_dtype(dtype))

    def test_gplinexpr(self):
        dtype = pandas_dtype("gplinexpr")
        self.assertTrue(is_extension_array_dtype(dtype))

    def test_gpquadexpr(self):
        dtype = pandas_dtype("gpquadexpr")
        self.assertTrue(is_extension_array_dtype(dtype))


class TestGurobiMObjectArrayDtypes(GurobiModelTestCase):
    # Constructor takes ownership of the object it is passed.
    # Returned .dtype is dependent on the type of the inner object.

    def test_var_dtype(self):
        mvar = self.model.addMVar((10,))
        arr = create_gurobimobjectarray(mvar)
        self.assertIsInstance(arr.dtype, GurobiVarDtype)
        self.assertEqual(len(arr), 10)

    def test_linexpr_dtype(self):
        x = self.model.addMVar((5,))
        arr = create_gurobimobjectarray(2.0 * x + 5)
        self.assertIsInstance(arr.dtype, GurobiLinExprDtype)
        self.assertEqual(len(arr), 5)

    def test_quadexpr_dtype(self):
        x = self.model.addMVar((7,))
        y = self.model.addMVar((7,))
        arr = create_gurobimobjectarray(2.0 * x * y + x + 5)
        self.assertIsInstance(arr.dtype, GurobiQuadExprDtype)
        self.assertEqual(len(arr), 7)


class TestGurobiMObjectArrayGetItem(GurobiModelTestCase):
    def setUp(self):
        super().setUp()
        self.mvar = self.model.addMVar((10,), name="x")
        self.arr = create_gurobimobjectarray(self.mvar)
        nan_mask = np.array([True, False] * 5).astype(bool)
        self.arr_nulls = GurobiMObjectArray(self.mvar, nan_mask)
        self.model.update()

    def test_scalar(self):
        # For scalar ``item``, return a scalar value suitable for the array's
        # type. This should be an instance of ``self.dtype.type``.
        for i in range(10):
            obj = self.arr[i]
            self.assertIsInstance(obj, gp.Var)
            self.assertEqual(obj.VarName, f"x[{i}]")

    def test_scalar_nulls(self):
        for i in range(10):
            obj = self.arr_nulls[i]
            if i % 2 == 0:
                self.assertIsNone(obj)
            else:
                self.assertIsInstance(obj, gp.Var)
                self.assertEqual(obj.VarName, f"x[{i}]")
                self.assertTrue(obj.sameAs(self.mvar[i].item()))

    def test_slice_1(self):
        # For slice ``key``, return an instance of ``ExtensionArray``, even
        # if the slice is length 0 or 1.
        for i in range(10):
            obj = self.arr[i : i + 1]
            self.assertIsInstance(obj, GurobiMObjectArray)
            self.assertEqual(len(obj), 1)
            self.assertEqual(obj[0].VarName, f"x[{i}]")

    def test_slice_1_nulls(self):
        # For slice ``key``, return an instance of ``ExtensionArray``, even
        # if the slice is length 0 or 1.
        for i in range(10):
            part = self.arr_nulls[i : i + 1]
            self.assertIsInstance(part, GurobiMObjectArray)
            self.assertEqual(len(part), 1)

            obj = part[0]
            if i % 2 == 0:
                self.assertIsNone(obj)
            else:
                self.assertIsInstance(obj, gp.Var)
                self.assertEqual(obj.VarName, f"x[{i}]")
                self.assertTrue(obj.sameAs(self.mvar[i].item()))

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

    def test_mask_nulls(self):
        # For a boolean mask, return an instance of ``ExtensionArray``, filtered
        # to the values where ``item`` is True.
        mask = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).astype(bool)
        obj = self.arr_nulls[mask]
        self.assertIsInstance(obj, GurobiMObjectArray)
        self.assertEqual(len(obj), 5)
        for i in range(5):
            if (2 * i + 1) % 2 == 0:
                self.assertIsNone(obj[i])
            else:
                self.assertEqual(obj[i].VarName, f"x[{2*i + 1}]")


class TestGurobiMObjectArrayTake(GurobiModelTestCase):
    def setUp(self):
        super().setUp()
        mvar = self.model.addMVar((8,), name="x")
        self.arr = create_gurobimobjectarray(mvar)
        self.model.update()

    def test_1(self):
        # Test normal positional indices
        indices = [1, 3, 4, 7]
        obj = self.arr.take(np.array(indices))
        self.assertIsInstance(obj, GurobiMObjectArray)
        self.assertEqual(len(obj), 4)
        for i in range(4):
            self.assertEqual(obj[i].VarName, f"x[{indices[i]}]")

    def test_2(self):
        # Test null fill
        indices = [1, 4, -1, 3]
        obj = self.arr.take(np.array(indices), allow_fill=True)
        self.assertIsInstance(obj, GurobiMObjectArray)
        self.assertEqual(len(obj), 4)

        self.assertEqual(obj[0].VarName, "x[1]")
        self.assertEqual(obj[1].VarName, "x[4]")
        self.assertIsNone(obj[2])
        self.assertEqual(obj[3].VarName, "x[3]")

    def test_3(self):
        # Test repeated null fill
        tmp = self.arr.take(np.array([1, 6, -1, 3, -1, 4]), allow_fill=True)
        obj = tmp.take(np.array([0, 1, 2, 1, -1, 4, -1, 0]), allow_fill=True)
        self.assertIsInstance(obj, GurobiMObjectArray)
        self.assertEqual(len(obj), 8)

        self.assertTrue(obj[0].sameAs(self.arr[1]))
        self.assertTrue(obj[1].sameAs(self.arr[6]))
        self.assertIsNone(obj[2])
        self.assertTrue(obj[3].sameAs(self.arr[6]))
        self.assertIsNone(obj[4])
        self.assertIsNone(obj[5])
        self.assertIsNone(obj[6])
        self.assertTrue(obj[7].sameAs(self.arr[1]))


class TestGurobiMObjectArrayCopy(GurobiModelTestCase):
    def test_1(self):
        mvar = self.model.addMVar((20,))
        arr = create_gurobimobjectarray(mvar)
        arr_copy = arr.copy()
        self.assertIsNot(arr, arr_copy)
        for i in range(20):
            self.assertTrue(arr[i].sameAs(arr_copy[i]))

    # TODO add more tests here after __iadd__ works


# Only minimal tests for arithmetic operators here. The array type just
# delegates to gurobi M* class operations, and more extensive testing is done on
# the resulting Series in test_operators.

# The exception is null handling ... checking that the mask is carried through
# properly is a pain ...


class TestGurobiMObjectArrayAdd(GurobiModelTestCase):
    def test_vararray_plus_scalar(self):
        vararr = create_gurobimobjectarray(self.model.addMVar((5,)))
        self.assertIsInstance(vararr.dtype, GurobiVarDtype)
        learr = vararr + 2.0
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(learr[i], vararr[i] + 2.0)

    def test_vararray(self):
        x = create_gurobimobjectarray(self.model.addMVar((5,)))
        y = create_gurobimobjectarray(self.model.addMVar((5,)))
        learr = x + y
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(learr[i], x[i] + y[i])

    def test_nan_handling_1(self):
        x = self.model.addMVar((3,))
        nan_mask = np.array([False, True, False])
        x_arr = create_gurobimobjectarray(x, nan_mask)
        res_arr = x_arr + 1.0

        self.assert_linexpr_equal(res_arr[0], x[0].item() + 1.0)
        self.assertIsNone(res_arr[1])
        self.assert_linexpr_equal(res_arr[2], x[2].item() + 1.0)

    def test_nan_handling_2(self):
        x = self.model.addMVar((5,))
        x_nan_mask = np.array([False, True, True, False, False])
        x_arr = create_gurobimobjectarray(x, x_nan_mask)

        y = self.model.addMVar((5,))
        y_nan_mask = np.array([False, False, True, False, True])
        y_arr = create_gurobimobjectarray(y, y_nan_mask)

        res_arr = x_arr + y_arr + 2.0

        self.assert_linexpr_equal(res_arr[0], x[0].item() + y[0].item() + 2.0)
        self.assertIsNone(res_arr[1])
        self.assertIsNone(res_arr[2])
        self.assert_linexpr_equal(res_arr[3], x[3].item() + y[3].item() + 2.0)
        self.assertIsNone(res_arr[4])


class TestGurobiMObjectArrayRadd(GurobiModelTestCase):
    def test_var_plus_vararray(self):
        vararr = create_gurobimobjectarray(self.model.addMVar((5,)))
        x = self.model.addVar()
        self.assertIsInstance(vararr.dtype, GurobiVarDtype)
        learr = x + vararr
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            # Term ordering is different as radd just delegates to add
            self.assert_linexpr_equal(learr[i], vararr[i] + x)

    def test_nan_handling_1(self):
        x = self.model.addMVar((3,))
        nan_mask = np.array([False, True, False])
        x_arr = create_gurobimobjectarray(x, nan_mask)
        res_arr = 1.0 + x_arr

        self.assert_linexpr_equal(res_arr[0], x[0].item() + 1.0)
        self.assertIsNone(res_arr[1])
        self.assert_linexpr_equal(res_arr[2], x[2].item() + 1.0)


class TestGurobiMObjectArrayIadd(GurobiModelTestCase):
    def test_scalar(self):
        mvar = self.model.addMVar((5,))
        arr = create_gurobimobjectarray(mvar)
        self.assertIsInstance(arr.dtype, GurobiVarDtype)
        ref = arr
        arr += 5.0
        self.assertIs(ref, arr)
        self.assertIsInstance(arr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(arr[i], mvar[i].item() + 5.0)

    def test_vararray(self):
        x = self.model.addMVar((5,))
        y = self.model.addMVar((5,))
        xarr = create_gurobimobjectarray(x)
        yarr = create_gurobimobjectarray(y)
        ref = xarr
        xarr += yarr
        self.assertIs(ref, xarr)
        self.assertIsInstance(xarr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(xarr[i], x[i].item() + y[i].item())

    def test_nan_handling_1(self):
        x = self.model.addMVar((3,))
        nan_mask = np.array([False, True, False])
        arr = create_gurobimobjectarray(x, nan_mask)
        arr += 1.0

        self.assert_linexpr_equal(arr[0], x[0].item() + 1.0)
        self.assertIsNone(arr[1])
        self.assert_linexpr_equal(arr[2], x[2].item() + 1.0)

    def test_nan_handling_2(self):
        x = self.model.addMVar((5,))
        x_nan_mask = np.array([False, True, True, False, False])
        arr = create_gurobimobjectarray(x, x_nan_mask)

        y = self.model.addMVar((5,))
        y_nan_mask = np.array([False, False, True, False, True])
        y_arr = create_gurobimobjectarray(y, y_nan_mask)

        arr += y_arr + 2.0

        self.assert_linexpr_equal(arr[0], x[0].item() + y[0].item() + 2.0)
        self.assertIsNone(arr[1])
        self.assertIsNone(arr[2])
        self.assert_linexpr_equal(arr[3], x[3].item() + y[3].item() + 2.0)
        self.assertIsNone(arr[4])


class TestGurobiMObjectArraySub(GurobiModelTestCase):
    def test_vararray_minus_linexpr(self):
        vararr = create_gurobimobjectarray(self.model.addMVar((5,)))
        x = self.model.addVar()
        self.assertIsInstance(vararr.dtype, GurobiVarDtype)
        learr = vararr - (x + 1.0)
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(learr[i], vararr[i] - x - 1.0)

    def test_vararray(self):
        x = create_gurobimobjectarray(self.model.addMVar((5,)))
        y = create_gurobimobjectarray(self.model.addMVar((5,)))
        learr = x - y
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(learr[i], x[i] - y[i])

    def test_nan_handling_1(self):
        x = self.model.addMVar((3,))
        nan_mask = np.array([False, True, False])
        x_arr = create_gurobimobjectarray(x, nan_mask)
        res_arr = x_arr - 1.0

        self.assert_linexpr_equal(res_arr[0], x[0].item() - 1.0)
        self.assertIsNone(res_arr[1])
        self.assert_linexpr_equal(res_arr[2], x[2].item() - 1.0)

    def test_nan_handling_2(self):
        x = self.model.addMVar((5,))
        x_nan_mask = np.array([False, False, True, True, False])
        x_arr = create_gurobimobjectarray(x, x_nan_mask)

        y = self.model.addMVar((5,))
        y_nan_mask = np.array([True, False, True, False, True])
        y_arr = create_gurobimobjectarray(y, y_nan_mask)

        res_arr = x_arr - (y_arr + 2.0)

        self.assertIsNone(res_arr[0])
        self.assert_linexpr_equal(res_arr[1], x[1].item() - y[1].item() - 2.0)
        self.assertIsNone(res_arr[2])
        self.assertIsNone(res_arr[3])
        self.assertIsNone(res_arr[4])


class TestGurobiMObjectArrayRsub(GurobiModelTestCase):
    def test_linexpr_minus_vararray(self):
        vararr = create_gurobimobjectarray(self.model.addMVar((5,)))
        x = self.model.addVar()
        self.assertIsInstance(vararr.dtype, GurobiVarDtype)
        learr = (2 * x + 4) - vararr
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            # Term ordering is different as radd just delegates to add
            self.assert_linexpr_equal(learr[i], -vararr[i] + 2 * x + 4)

    def test_nan_handling_1(self):
        x = self.model.addMVar((3,))
        nan_mask = np.array([False, False, True])
        x_arr = create_gurobimobjectarray(x, nan_mask)
        res_arr = 1.0 - x_arr

        self.assert_linexpr_equal(res_arr[0], 1.0 - x[0].item())
        self.assert_linexpr_equal(res_arr[1], 1.0 - x[1].item())
        self.assertIsNone(res_arr[2])


class TestGurobiMObjectArrayIsub(GurobiModelTestCase):
    def test_mvar_0d(self):
        mvar = self.model.addMVar((5,))
        arr = create_gurobimobjectarray(mvar)
        self.assertIsInstance(arr.dtype, GurobiVarDtype)
        ref = arr
        arr -= mvar[2]
        self.assertIs(ref, arr)
        self.assertIsInstance(arr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(arr[i], mvar[i].item() - mvar[2].item())

    def test_vararray(self):
        x = self.model.addMVar((5,))
        y = self.model.addMVar((5,))
        xarr = create_gurobimobjectarray(x)
        yarr = create_gurobimobjectarray(y)
        ref = xarr
        xarr -= yarr
        self.assertIs(ref, xarr)
        self.assertIsInstance(xarr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(xarr[i], x[i].item() - y[i].item())

    def test_nan_handling_1(self):
        x = self.model.addMVar((3,))
        nan_mask = np.array([True, True, False])
        arr = create_gurobimobjectarray(x, nan_mask)
        arr -= 1.0

        self.assertIsNone(arr[0])
        self.assertIsNone(arr[1])
        self.assert_linexpr_equal(arr[2], x[2].item() - 1.0)

    def test_nan_handling_2(self):
        x = self.model.addMVar((5,))
        x_nan_mask = np.array([True, False, True, False, False])
        arr = create_gurobimobjectarray(x, x_nan_mask)

        y = self.model.addMVar((5,))
        y_nan_mask = np.array([False, False, True, False, True])
        y_arr = create_gurobimobjectarray(y, y_nan_mask)

        arr -= y_arr + 2.0

        self.assertIsNone(arr[0])
        self.assert_linexpr_equal(arr[1], x[1].item() - y[1].item() - 2.0)
        self.assertIsNone(arr[2])
        self.assert_linexpr_equal(arr[3], x[3].item() - y[3].item() - 2.0)
        self.assertIsNone(arr[4])


class TestGurobiMObjectArrayMul(GurobiModelTestCase):
    def test_vararray_times_scalar(self):
        vararr = create_gurobimobjectarray(self.model.addMVar((5,)))
        self.assertIsInstance(vararr.dtype, GurobiVarDtype)
        learr = vararr * 2.0
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(learr[i], vararr[i] * 2.0)

    def test_vararray_times_array(self):
        vararr = create_gurobimobjectarray(self.model.addMVar((5,)))
        a = np.arange(1, 6)
        self.assertIsInstance(vararr.dtype, GurobiVarDtype)
        learr = vararr * a
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(learr[i], vararr[i] * (i + 1))

    def test_vararray_times_vararray(self):
        x = create_gurobimobjectarray(self.model.addMVar((5,)))
        y = create_gurobimobjectarray(self.model.addMVar((5,)))
        self.assertIsInstance(x.dtype, GurobiVarDtype)
        self.assertIsInstance(y.dtype, GurobiVarDtype)
        qearr = x * y
        self.assertIsInstance(qearr.dtype, GurobiQuadExprDtype)
        for i in range(5):
            self.assert_quadexpr_equal(qearr[i], x[i] * y[i])

    def test_nan_handling_1(self):
        x = self.model.addMVar((3,))
        y = self.model.addVar()
        nan_mask = np.array([False, True, False])
        x_arr = create_gurobimobjectarray(x, nan_mask)
        res_arr = x_arr * (y + 1)

        self.assert_quadexpr_equal(res_arr[0], x[0].item() * y + x[0].item())
        self.assertIsNone(res_arr[1])
        self.assert_quadexpr_equal(res_arr[2], x[2].item() * y + x[2].item())

    def test_nan_handling_2(self):
        x = self.model.addMVar((5,))
        x_nan_mask = np.array([False, False, True, False, False])
        x_arr = create_gurobimobjectarray(x, x_nan_mask)

        y = self.model.addMVar((5,))
        y_nan_mask = np.array([False, True, True, False, True])
        y_arr = create_gurobimobjectarray(y, y_nan_mask)

        res_arr = x_arr * (2.0 * y_arr + 3.0)

        self.assert_quadexpr_equal(
            res_arr[0], 2.0 * x[0].item() * y[0].item() + 3.0 * x[0].item()
        )
        self.assertIsNone(res_arr[1])
        self.assertIsNone(res_arr[2])
        self.assert_quadexpr_equal(
            res_arr[3], 2.0 * x[3].item() * y[3].item() + 3.0 * x[3].item()
        )
        self.assertIsNone(res_arr[4])


class TestGurobiMObjectArrayRmul(GurobiModelTestCase):
    def test_vararray_times_scalar(self):
        vararr = create_gurobimobjectarray(self.model.addMVar((5,)))
        self.assertIsInstance(vararr.dtype, GurobiVarDtype)
        learr = 2.0 * vararr
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(learr[i], vararr[i] * 2.0)

    def test_vararray_times_array(self):
        vararr = create_gurobimobjectarray(self.model.addMVar((5,)))
        a = np.arange(1, 6)
        self.assertIsInstance(vararr.dtype, GurobiVarDtype)
        learr = a * vararr
        self.assertIsInstance(learr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(learr[i], vararr[i] * (i + 1))

    def test_nan_handling_1(self):
        x = self.model.addMVar((3,))
        y = self.model.addVar()
        nan_mask = np.array([False, True, False])
        x_arr = create_gurobimobjectarray(x, nan_mask)
        res_arr = (2.0 * y) * x_arr

        self.assert_quadexpr_equal(res_arr[0], 2.0 * x[0].item() * y)
        self.assertIsNone(res_arr[1])
        self.assert_quadexpr_equal(res_arr[2], 2.0 * x[2].item() * y)


class TestGurobiMObjectArrayImul(GurobiModelTestCase):
    def test_vararray_times_scalar(self):
        x = self.model.addMVar((5,))
        arr = create_gurobimobjectarray(x)
        self.assertIsInstance(arr.dtype, GurobiVarDtype)
        ref = arr
        arr *= 2.0
        self.assertIs(ref, arr)
        self.assertIsInstance(arr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(arr[i], x[i].item() * 2.0)

    def test_vararray_times_array(self):
        x = self.model.addMVar((5,))
        arr = create_gurobimobjectarray(x)
        self.assertIsInstance(arr.dtype, GurobiVarDtype)
        a = np.arange(1, 6)
        ref = arr
        arr *= a
        self.assertIs(ref, arr)
        self.assertIsInstance(arr.dtype, GurobiLinExprDtype)
        for i in range(5):
            self.assert_linexpr_equal(arr[i], x[i].item() * (i + 1))

    def test_nan_handling_1(self):
        x = self.model.addMVar((3,))
        y = self.model.addVar()
        nan_mask = np.array([False, True, False])
        arr = create_gurobimobjectarray(x, nan_mask)
        arr *= y + 1

        self.assert_quadexpr_equal(arr[0], x[0].item() * y + x[0].item())
        self.assertIsNone(arr[1])
        self.assert_quadexpr_equal(arr[2], x[2].item() * y + x[2].item())

    def test_nan_handling_2(self):
        x = self.model.addMVar((5,))
        x_nan_mask = np.array([False, False, True, False, False])
        arr = create_gurobimobjectarray(x, x_nan_mask)

        y = self.model.addMVar((5,))
        y_nan_mask = np.array([False, True, True, False, True])
        y_arr = create_gurobimobjectarray(y, y_nan_mask)

        arr *= 2.0 * y_arr + 3.0

        self.assert_quadexpr_equal(
            arr[0], 2.0 * x[0].item() * y[0].item() + 3.0 * x[0].item()
        )
        self.assertIsNone(arr[1])
        self.assertIsNone(arr[2])
        self.assert_quadexpr_equal(
            arr[3], 2.0 * x[3].item() * y[3].item() + 3.0 * x[3].item()
        )
        self.assertIsNone(arr[4])


class TestGurobiMObjectArrayPow(GurobiModelTestCase):
    def test_power_2(self):
        x = create_gurobimobjectarray(self.model.addMVar((7,)))
        arr = x**2
        self.assertIsInstance(arr.dtype, GurobiQuadExprDtype)
        for i in range(7):
            self.assert_quadexpr_equal(x[i] ** 2, arr[i])

    def test_nan_handling_1(self):
        x = self.model.addMVar((5,))
        nan_mask = np.array([False, False, True, True, False])
        x_arr = create_gurobimobjectarray(x, nan_mask)
        res_arr = x_arr**2

        for i in [0, 1, 4]:
            self.assert_quadexpr_equal(res_arr[i], x[i].item() ** 2)
        for i in [2, 3]:
            self.assertIsNone(res_arr[i])
