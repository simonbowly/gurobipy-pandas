import unittest

import gurobipy as gp


def sorted_linear_terms(linexpr):
    terms = [(linexpr.getVar(i), linexpr.getCoeff(i)) for i in range(linexpr.size())]
    return sorted(terms, key=lambda term: (id(term[0]), term[1]))


class GurobiModelTestCase(unittest.TestCase):
    def setUp(self):
        self.env = gp.Env()
        self.model = gp.Model(env=self.env)

    def tearDown(self):
        self.model.close()
        self.env.close()

    def assert_expression_equal(self, expr1, expr2):
        if isinstance(expr1, gp.LinExpr):
            self.assert_linexpr_equal(expr1, expr2)
        elif isinstance(expr1, gp.QuadExpr):
            self.assert_quadexpr_equal(expr1, expr2)
        else:
            self.fail("Bad type passed to assert_expression_equal_unordered")

    def assert_linexpr_equal(self, expr1, expr2):
        self.assertIsInstance(expr1, gp.LinExpr)
        self.assertIsInstance(expr2, gp.LinExpr)
        self.assertEqual(expr1.getConstant(), expr2.getConstant())
        self.assertEqual(expr1.size(), expr2.size())
        for i in range(expr1.size()):
            self.assertTrue(expr1.getVar(i).sameAs(expr2.getVar(i)))
            self.assertEqual(expr1.getCoeff(i), expr2.getCoeff(i))

    def assert_quadexpr_equal(self, expr1, expr2):
        self.assertIsInstance(expr1, gp.QuadExpr)
        self.assertIsInstance(expr2, gp.QuadExpr)
        self.assert_linexpr_equal(expr1.getLinExpr(), expr2.getLinExpr())
        self.assertEqual(expr1.size(), expr2.size())
        for i in range(expr1.size()):
            self.assertTrue(expr1.getVar1(i).sameAs(expr2.getVar1(i)))
            self.assertTrue(expr1.getVar2(i).sameAs(expr2.getVar2(i)))
            self.assertEqual(expr1.getCoeff(i), expr2.getCoeff(i))

    def assert_expression_equal_unordered(self, expr1, expr2):
        if isinstance(expr1, gp.LinExpr):
            self.assert_linexpr_equal_unordered(expr1, expr2)
        else:
            self.fail("Bad type passed to assert_expression_equal_unordered")

    def assert_linexpr_equal_unordered(self, expr1, expr2):
        # Order-independent (no grouping of like terms)
        self.assertIsInstance(expr1, gp.LinExpr)
        self.assertIsInstance(expr2, gp.LinExpr)
        self.assertEqual(expr1.getConstant(), expr2.getConstant())
        self.assertEqual(expr1.size(), expr2.size())
        for (var1, coeff1), (var2, coeff2) in zip(
            sorted_linear_terms(expr1), sorted_linear_terms(expr2)
        ):
            self.assertTrue(var1.sameAs(var2))
            self.assertEqual(coeff1, coeff2)
