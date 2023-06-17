import gurobipy as gp

from pandas.api.extensions import (
    ExtensionDtype,
    ExtensionArray,
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
