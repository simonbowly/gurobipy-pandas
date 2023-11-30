import gurobipy as gp
import numpy as np
import scipy.sparse as sp
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
    def __init__(self, mobj, nan_mask, lazy_ops=None):
        assert isinstance(mobj, (gp.MVar, gp.MLinExpr, gp.MQuadExpr))
        assert mobj.ndim == 1
        assert nan_mask.dtype == bool
        # assert nan_mask.shape == mobj.shape  # not true with lazy eval ...
        self.mobj = mobj
        self.nan_mask = nan_mask
        if lazy_ops is None:
            self.lazy_ops = []
        else:
            self.lazy_ops = lazy_ops

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

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if any(isinstance(s, gp.QuadExpr) for s in scalars):
            mobj = gp.MQuadExpr.zeros(len(scalars))
            for i, s in enumerate(scalars):
                mobj[i] += s
        elif any(isinstance(s, gp.LinExpr) for s in scalars):
            mobj = gp.MLinExpr.zeros(len(scalars))
            for i, s in enumerate(scalars):
                mobj[i] += s
        else:
            scalars = scalars.tolist()
            indices = scalars[0]["take"]
            indptr = [s["slice"].start for s in scalars]
            indptr.append(len(indices))
            data = np.ones(indices.shape[0])
            A = sp.csr_array((data, indices, indptr))
            mobj = A @ scalars[0]["mobj"]
        nan_mask = np.zeros(mobj.shape, dtype=bool)
        return GurobiMObjectArray(mobj, nan_mask)

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
        elif isinstance(item, slice) and item.step is None:
            return GurobiMObjectArray(
                self.mobj,
                self.nan_mask[item],
                lazy_ops=self.lazy_ops + [("slice", item)],
            )
        else:
            # Return slice/mask of both the expression and the mask
            # TODO does the expression ever need to be copied?
            return GurobiMObjectArray(self.mobj[item], self.nan_mask[item])

    def take(self, indices, allow_fill=False, fill_value=None):
        assert fill_value is None or fill_value is np.nan
        if allow_fill:
            nan_mask = self.nan_mask[indices] | (indices == -1)
        else:
            nan_mask = self.nan_mask[indices]
        if isinstance(indices, np.ndarray) and len(indices) == len(self):
            return GurobiMObjectArray(
                self.mobj, nan_mask, lazy_ops=self.lazy_ops + [("take", indices)]
            )
        mobj = self.mobj[indices]  # TODO does this need a copy?
        return GurobiMObjectArray(mobj, nan_mask)

    def copy(self):
        return GurobiMObjectArray(self.mobj.copy(), self.nan_mask.copy())

    def isna(self):
        return self.nan_mask.copy()

    def _reduce(self, name, skipna=True, **kwargs):
        assert name == "sum"
        if self.lazy_ops:
            # Lazily evaluate the groupby case
            assert len(self.lazy_ops) == 2
            (take_op, take_indices), (slice_op, slice_item) = self.lazy_ops
            assert take_op == "take" and slice_op == "slice"
            assert (
                isinstance(take_indices, np.ndarray)
                and take_indices.shape == self.mobj.shape
            )
            assert isinstance(slice_item, slice) and slice_item.step is None
            return {
                "take": take_indices,
                "slice": slice_item,
                "mobj": self.mobj,
                "op": name,
            }
        else:
            return self.mobj.sum().item()

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
        import pdb

        pdb.set_trace()
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
