import unittest

import gurobipy as gp
import pandas as pd
from pandas.testing import assert_index_equal

import gurobipy_pandas as gppd

from .utils import GurobiModelTestCase


class TestGroupBySum(GurobiModelTestCase):
    def test_series_1(self):
        index = pd.MultiIndex.from_product([(0, 1), (0, 1)], names=["i", "j"])
        x = gppd.add_vars(self.model, index, name="x")
        self.model.update()

        group_i = x.groupby("i").sum()
        assert_index_equal(group_i.index, pd.RangeIndex(2, name="i"))
        self.assert_linexpr_equal(group_i.loc[0], x[0, 0] + x[0, 1])
        self.assert_linexpr_equal(group_i.loc[1], x[1, 0] + x[1, 1])

        gppd.add_constrs(self.model, group_i, "=", 1)

    def test_series_2(self):
        index = pd.MultiIndex.from_product([(0, 1), (0, 1)], names=["i", "j"])
        x = gppd.add_vars(self.model, index, name="x")
        self.model.update()

        group_j = x.groupby("j").sum()
        assert_index_equal(group_j.index, pd.RangeIndex(2, name="j"))
        self.assert_linexpr_equal(group_j.loc[0], x[0, 0] + x[1, 0])
        self.assert_linexpr_equal(group_j.loc[1], x[0, 1] + x[1, 1])

        gppd.add_constrs(self.model, group_j, "=", 1)
