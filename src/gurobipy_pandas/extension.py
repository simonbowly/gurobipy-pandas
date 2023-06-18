import gurobipy as gp
import numpy as np
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_extension_dtype,
)


@register_extension_dtype
class GurobiVarDtype(ExtensionDtype):
    name = "gpvar"
    type = gp.Var  # scalar type returned from indexing
    kind = "O"

    @classmethod
    def construct_array_type(cls):
        return GurobiMObjectArray


@register_extension_dtype
class GurobiLinExprDtype(ExtensionDtype):
    name = "gplinexpr"
    type = gp.LinExpr  # scalar type returned from indexing
    kind = "O"

    @classmethod
    def construct_array_type(cls):
        return GurobiMObjectArray


@register_extension_dtype
class GurobiQuadExprDtype(ExtensionDtype):
    name = "gpquadexpr"
    type = gp.QuadExpr  # scalar type returned from indexing
    kind = "O"

    @classmethod
    def construct_array_type(cls):
        return GurobiMObjectArray


class GurobiMObjectArray(ExtensionArray):
    def __init__(self, mobj, nan_mask):
        assert isinstance(mobj, (gp.MVar, gp.MLinExpr, gp.MQuadExpr))
        assert mobj.ndim == 1
        assert np.dtype(bool) == nan_mask.dtype
        assert nan_mask.shape == mobj.shape
        self.mobj = mobj
        self.nan_mask = nan_mask

    def __len__(self):
        return self.mobj.size

    @property
    def dtype(self) -> ExtensionDtype:
        if isinstance(self.mobj, gp.MVar):
            return GurobiVarDtype()
        elif isinstance(self.mobj, gp.MLinExpr):
            return GurobiLinExprDtype()
        elif isinstance(self.mobj, gp.MQuadExpr):
            return GurobiQuadExprDtype()

    def __getitem__(self, item):
        """
        Pandas has clear guidance here in ExtensionDType:

        - For scalar ``item``, return a scalar value suitable for the array's
          type. This should be an instance of ``self.dtype.type``.
        - For slice ``key``, return an instance of ``ExtensionArray``, even if
          the slice is length 0 or 1.
        - For a boolean mask, return an instance of ``ExtensionArray``, filtered
          to the values where ``item`` is True.
        """
        if isinstance(item, int):
            # Scalar: return a python object
            if self.nan_mask[item]:
                return None
            return self.mobj[item].item()
        else:
            # Return slice/mask of both the expression and the mask
            # TODO does the expression ever need to be copied?
            return GurobiMObjectArray(self.mobj[item], self.nan_mask[item])

    def take(self, indices, allow_fill=False, fill_value=None):
        assert fill_value is None
        if allow_fill:
            nan_mask = self.nan_mask[indices] | (indices == -1)
        else:
            nan_mask = self.nan_mask[indices]
        mobj = self.mobj[indices]  # TODO does this need a copy?
        return GurobiMObjectArray(mobj, nan_mask)

    def copy(self):
        return GurobiMObjectArray(self.mobj.copy(), self.nan_mask.copy())

    def isna(self):
        return self.nan_mask.copy()

    def _prepare_operands(self, other, mul=False):
        # Return an operand which can work with self.mobj, and the correct
        # nan mask for the result.
        if isinstance(other, GurobiMObjectArray):
            nan_mask = self.nan_mask | other.nan_mask
            other = other.mobj
        else:
            if mul and isinstance(other, gp.LinExpr):
                # Workaround a missing operator implementation in gurobipy <=
                # 10.0.2. Convert LinExpr to 0d MLinExpr before passing to
                # multiply operations.
                other = gp.MLinExpr.zeros(tuple()) + other
            nan_mask = self.nan_mask
        return other, nan_mask

    def __add__(self, other):
        other, nan_mask = self._prepare_operands(other)
        return GurobiMObjectArray(self.mobj + other, nan_mask)

    def __radd__(self, other):
        other, nan_mask = self._prepare_operands(other)
        return GurobiMObjectArray(other + self.mobj, nan_mask)

    def __iadd__(self, other):
        other, nan_mask = self._prepare_operands(other)
        self.mobj += other
        self.nan_mask = nan_mask
        return self

    def __sub__(self, other):
        other, nan_mask = self._prepare_operands(other)
        return GurobiMObjectArray(self.mobj - other, nan_mask)

    def __rsub__(self, other):
        other, nan_mask = self._prepare_operands(other)
        return GurobiMObjectArray(other - self.mobj, nan_mask)

    def __isub__(self, other):
        other, nan_mask = self._prepare_operands(other)
        self.mobj -= other
        self.nan_mask = nan_mask
        return self

    def __mul__(self, other):
        other, nan_mask = self._prepare_operands(other, mul=True)
        return GurobiMObjectArray(self.mobj * other, nan_mask)

    def __rmul__(self, other):
        other, nan_mask = self._prepare_operands(other, mul=True)
        return GurobiMObjectArray(self.mobj * other, nan_mask)

    def __imul__(self, other):
        other, nan_mask = self._prepare_operands(other, mul=True)
        self.mobj *= other
        self.nan_mask = nan_mask
        return self

    def __pow__(self, power):
        power, nan_mask = self._prepare_operands(power)
        return GurobiMObjectArray(self.mobj**power, nan_mask)
