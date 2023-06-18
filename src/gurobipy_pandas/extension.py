import gurobipy as gp
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


class GurobiMObjectArray(ExtensionArray):
    def __init__(self, mobj):
        assert isinstance(mobj, (gp.MVar, gp.MLinExpr))
        assert mobj.ndim == 1
        self.mobj = mobj

    def __len__(self):
        return self.mobj.size

    @property
    def dtype(self) -> ExtensionDtype:
        if isinstance(self.mobj, gp.MVar):
            return GurobiVarDtype()
        elif isinstance(self.mobj, gp.MLinExpr):
            return GurobiLinExprDtype()

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
            return self.mobj[item].item()
        return GurobiMObjectArray(self.mobj[item])  # TODO does this need a copy?

    def take(self, indices, allow_fill=False, fill_value=None):
        return GurobiMObjectArray(self.mobj[indices])  # TODO does this need a copy?

    def copy(self):
        return GurobiMObjectArray(self.mobj.copy())

    def __add__(self, other):
        return GurobiMObjectArray(self.mobj + other)

    def __radd__(self, other):
        return GurobiMObjectArray(self.mobj + other)
