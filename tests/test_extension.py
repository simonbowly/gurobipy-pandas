import unittest

from pandas.api.types import pandas_dtype, is_extension_array_dtype

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
