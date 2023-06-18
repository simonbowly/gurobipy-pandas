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
        return GurobiVarArray


class GurobiVarArray(ExtensionArray):
    def __init__(self, mvar):
        assert isinstance(mvar, gp.MVar)
        assert mvar.ndim == 1
        self.mvar = mvar

    def __len__(self):
        return self.mvar.size

    @property
    def dtype(self) -> ExtensionDtype:
        return GurobiVarDtype()

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
            return self.mvar[item].item()
        return GurobiVarArray(self.mvar[item])  # TODO does this need a copy?

    def take(self, indices, allow_fill=False, fill_value=None):
        return GurobiVarArray(self.mvar[indices])
