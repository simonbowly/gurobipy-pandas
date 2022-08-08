""" Tests for arithmetic operations. Intention is to verify that gurobipy
objects correctly defer to Series for arithmetic operations. """

import operator
import unittest

import gurobipy_pandas
import pandas as pd

from .utils import GurobiTestCase


class TestAdd(GurobiTestCase):

    op = operator.add

    def test_varseries_var(self):
        x = pd.RangeIndex(5).grb.pd_add_vars(self.model, name="x")
        y = self.model.addVar(name="y")
        self.model.update()
        result = self.op(x, y)
        self.assertIsInstance(result, pd.Series)
        for i in range(5):
            self.assert_expression_equal(result[i], self.op(x[i], y))

    @unittest.expectedFailure
    def test_var_varseries(self):
        x = pd.RangeIndex(5).grb.pd_add_vars(self.model, name="x")
        y = self.model.addVar(name="y")
        self.model.update()
        result = self.op(y, x)
        self.assertIsInstance(result, pd.Series)
        for i in range(5):
            self.assert_expression_equal(result[i], self.op(y, x[i]))

    def test_dataseries_var(self):
        x = self.model.addVar(name="x")
        self.model.update()
        result = self.op(pd.Series(list(range(5))), x)
        self.assertIsInstance(result, pd.Series)
        # Note if we use extension types in future, this would not
        # come out as an extension type.
        for i in range(5):
            self.assert_expression_equal(result[i], self.op(i, x))

    @unittest.expectedFailure
    def test_var_dataseries(self):
        x = self.model.addVar(name="x")
        self.model.update()
        result = self.op(x, pd.Series(list(range(5))))
        self.assertIsInstance(result, pd.Series)
        for i in range(5):
            self.assert_expression_equal(result[i], self.op(x, i))

    def test_varseries_linexpr(self):
        x = pd.RangeIndex(5).grb.pd_add_vars(self.model, name="x")
        y = self.model.addVar(name="y")
        le = 2 * y + 1
        self.model.update()
        result = self.op(x, le)
        self.assertIsInstance(result, pd.Series)
        for i in range(5):
            self.assert_expression_equal(result[i], self.op(x[i], +2 * y + 1))

    @unittest.expectedFailure
    def test_linexpr_varseries(self):
        x = pd.RangeIndex(5).grb.pd_add_vars(self.model, name="x")
        y = self.model.addVar(name="y")
        le = 2 * y + 1
        self.model.update()
        result = self.op(le, x)
        self.assertIsInstance(result, pd.Series)
        for i in range(5):
            self.assert_expression_equal(result[i], self.op(2 * y + 1, x[i]))

    def test_varseries_quadexpr(self):
        x = pd.RangeIndex(5).grb.pd_add_vars(self.model, name="x")
        y = self.model.addVar(name="y")
        qe = y * y + 2 * y + 3
        self.model.update()
        result = self.op(x, qe)
        self.assertIsInstance(result, pd.Series)
        for i in range(5):
            self.assert_expression_equal(result[i], self.op(x[i], y * y + 2 * y + 3))

    @unittest.expectedFailure
    def test_quadexpr_varseries(self):
        x = pd.RangeIndex(5).grb.pd_add_vars(self.model, name="x")
        y = self.model.addVar(name="y")
        qe = y * y + 2 * y + 3
        self.model.update()
        result = self.op(qe, x)
        self.assertIsInstance(result, pd.Series)
        for i in range(5):
            self.assert_expression_equal(result[i], self.op(y * y + 2 * y + 3, x[i]))


class TestSub(TestAdd):

    # Replace operator in TestAdd for sub tests
    op = operator.sub


class TestIadd(GurobiTestCase):

    op = operator.iadd
    checkop = operator.add

    def test_varseries_var(self):
        # Make series manually here so original isn't changed.
        x0 = list(self.model.addVars(3).values())
        x = pd.Series(x0)
        y = self.model.addVar(name="y")
        self.model.update()
        self.op(x, y)
        self.assertIsInstance(x, pd.Series)
        for i in range(3):
            self.assert_expression_equal(x[i], self.checkop(x0[i], y))

    def test_linexprseries_var(self):
        x = pd.RangeIndex(3).grb.pd_add_vars(self.model, name="x")
        le = x * 1
        y = self.model.addVar(name="y")
        self.model.update()
        self.op(le, y)
        self.assertIsInstance(le, pd.Series)
        for i in range(3):
            self.assert_expression_equal(le[i], self.checkop(x[i], y))

    def test_quadexprseries_var(self):
        x = pd.RangeIndex(3).grb.pd_add_vars(self.model, name="x")
        qe = x * x
        y = self.model.addVar(name="y")
        self.model.update()
        self.op(qe, y)
        self.assertIsInstance(qe, pd.Series)
        for i in range(3):
            self.assert_expression_equal(qe[i], self.checkop(x[i] * x[i], y))

    @unittest.expectedFailure
    def test_var_varseries(self):
        # Type uplift to Series (reassignment)
        x = pd.RangeIndex(3).grb.pd_add_vars(self.model, name="x")
        y0 = self.model.addVar(name="y")
        y = y0
        self.model.update()
        self.op(y, x)
        self.assertIsInstance(y, pd.Series)
        for i in range(3):
            self.assert_expression_equal(y[i], self.checkop(y0, x[i]))

    @unittest.expectedFailure
    def test_linexpr_varseries(self):
        # Type uplift to Series (reassignment)
        # Expected call chain:
        #   1. LinExpr.__iadd__(Series) -> return NotImplemented
        #   2. Series.__radd__(LinExpr) -> expanded along series
        #   3. LinExpr.__add__(Var) -> new LinExpr for each in series
        #   4. New series reassigned over le
        x = pd.RangeIndex(3).grb.pd_add_vars(self.model, name="x")
        y = self.model.addVar(name="y")
        le = 2 * y
        self.model.update()
        self.op(le, x)
        self.assertIsInstance(le, pd.Series)
        for i in range(3):
            self.assert_expression_equal(le[i], self.checkop(2 * y, x[i]))

    @unittest.expectedFailure
    def test_quadexpr_varseries(self):
        # Type uplift to Series (reassignment)
        x = pd.RangeIndex(3).grb.pd_add_vars(self.model, name="x")
        y = self.model.addVar(name="y")
        qe = y * y
        self.model.update()
        self.op(qe, x)
        self.assertIsInstance(qe, pd.Series)
        for i in range(3):
            self.assert_expression_equal(qe[i], self.checkop(y * y, x[i]))


class TestIsub(TestIadd):

    # Replace operator in TestIadd for isub tests
    op = operator.isub
    checkop = operator.sub


class TestMul(GurobiTestCase):
    def test_varseries_var(self):
        x = pd.RangeIndex(5).grb.pd_add_vars(self.model, name="x")
        y = self.model.addVar(name="y")
        self.model.update()
        result = x * y
        self.assertIsInstance(result, pd.Series)
        for i in range(5):
            self.assert_expression_equal(result[i], x[i] * y)

    @unittest.expectedFailure
    def test_var_varseries(self):
        x = pd.RangeIndex(5).grb.pd_add_vars(self.model, name="x")
        y = self.model.addVar(name="y")
        self.model.update()
        result = y * x
        self.assertIsInstance(result, pd.Series)
        for i in range(5):
            self.assert_expression_equal(result[i], y * x[i])

    def test_varseries_linexpr(self):
        x = pd.RangeIndex(5).grb.pd_add_vars(self.model, name="x")
        y = self.model.addVar(name="y")
        le = 2 * y
        self.model.update()
        result = x * le
        self.assertIsInstance(result, pd.Series)
        for i in range(5):
            self.assert_expression_equal(result[i], x[i] * 2 * y)

    @unittest.expectedFailure
    def test_linexpr_varseries(self):
        x = pd.RangeIndex(5).grb.pd_add_vars(self.model, name="x")
        y = self.model.addVar(name="y")
        le = 2 * y
        self.model.update()
        result = le * x
        self.assertIsInstance(result, pd.Series)
        for i in range(5):
            self.assert_expression_equal(result[i], 2 * y * x[i])

    def test_dataseries_quadexpr(self):
        s = pd.Series(list(range(5)))
        y = self.model.addVar(name="y")
        qe = y * y
        self.model.update()
        result = s * qe
        self.assertIsInstance(result, pd.Series)
        for i in range(5):
            self.assert_expression_equal(result[i], i * y * y)

    @unittest.expectedFailure
    def test_quadexpr_dataseries(self):
        s = pd.Series(list(range(5)))
        y = self.model.addVar(name="y")
        qe = y * y
        self.model.update()
        result = qe * s
        self.assertIsInstance(result, pd.Series)
        for i in range(5):
            self.assert_expression_equal(result[i], i * y * y)

    @unittest.expectedFailure
    def test_varseries_quadexpr(self):
        # Cannot multiply, should get a python TypeError
        x = pd.RangeIndex(5).grb.pd_add_vars(self.model, name="x")
        y = self.model.addVar(name="y")
        qe = y * y
        self.model.update()
        with self.assertRaises(TypeError):
            x * qe