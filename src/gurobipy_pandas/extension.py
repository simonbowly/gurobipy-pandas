import gurobipy as gp

from pandas.api.extensions import (
    ExtensionDtype,
    ExtensionArray,
    register_extension_dtype,
)


@register_extension_dtype
class GurobiVarDtype(ExtensionDtype):

    name = "gpvar"
    type = gp.Var
    kind = "O"

    @classmethod
    def construct_array_type(cls):
        return GurobiVarArray


class GurobiVarArray(ExtensionArray):
    @property
    def dtype(self) -> ExtensionDtype:
        return GurobiVarDtype()
