import pathlib

import gurobipy as gp
import numpy as np
import pandas as pd
import scipy.sparse as sp

import gurobipy_pandas as gppd
from gurobipy_pandas.util import gppd_global_options

here = pathlib.Path(__file__).parent


class AddVariables:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        self.env = gp.Env()
        self.model = gp.Model(env=self.env)
        self.index = pd.RangeIndex(2**16)
        self.frame = pd.DataFrame(
            [{"i": i, "j": j, "ub": 1.0} for i in range(2**8) for j in range(2**8)]
        ).set_index(["i", "j"])
        gppd_global_options["use_extension"] = False

    def teardown(self):
        self.model.close()
        self.env.close()

    def time_add_vars_series(self):
        gppd.add_vars(self.model, self.index)

    def time_add_vars_series_named(self):
        gppd.add_vars(self.model, self.index, name="x")

    def time_add_vars_frame(self):
        self.frame.gppd.add_vars(self.model, name="x")


class AddVariablesExtension(AddVariables):
    def setup(self):
        super().setup()
        gppd_global_options["use_extension"] = True


class NetflowModel:
    def setup(self):
        E = 2**14
        N = 2**10
        D = 2**6

        rng = np.random.default_rng(123)

        arr = rng.choice(np.arange(N), size=D, replace=False)
        arr.sort()
        self.node_data = pd.DataFrame(
            {"node": arr, "demand": rng.integers(-2, 2, size=arr.size)}
        ).set_index(["node"])

        arr = rng.choice(np.arange(N * N), size=E, replace=False)
        arr.sort()
        self.edge_data = pd.DataFrame(
            {
                "from": arr // N,
                "to": arr % N,
                "capacity": rng.integers(1, 4, size=arr.size),
                "cost": rng.integers(1, 4, size=arr.size),
            }
        ).set_index(["from", "to"])

        gppd_global_options["use_extension"] = False

    def time_build(self):
        with gp.Env() as env, gp.Model(env=env) as model:
            flow = gppd.add_vars(
                model, self.edge_data, ub="capacity", obj="cost", name="flow"
            )
            constrs = (
                pd.DataFrame(
                    {
                        "inflow": flow.groupby("to").agg(gp.quicksum),
                        "outflow": flow.groupby("from").agg(gp.quicksum),
                        "demand": self.node_data["demand"],
                    }
                )
                .fillna(0.0)
                .gppd.add_constrs(model, "inflow - outflow == demand", name="balance")
            )
            model.update()
            print(
                f"{model.NumVars=} {model.NumConstrs=} {model.NumNZs=} {model.FingerPrint=}"
            )
            model.write(f"{here}/{self.__class__.__name__}.lp")


class NetflowModelExtension(NetflowModel):
    def setup(self):
        super().setup()
        gppd_global_options["use_extension"] = True


class NetflowModelManual(NetflowModel):
    def time_build(self):
        edge_data = self.edge_data.reset_index()
        nrows = max(edge_data["from"].max(), edge_data["to"].max()) + 1
        ncols = self.edge_data.shape[0]

        indices = edge_data["from"].values
        indptr = np.arange(indices.shape[0] + 1)
        data = np.ones(indices.shape)
        A_out = sp.csc_array((data, indices, indptr), shape=(nrows, ncols)).tocsc()

        indices = edge_data["to"].values
        indptr = np.arange(indices.shape[0] + 1)
        data = np.ones(indices.shape)
        A_in = sp.csc_array((data, indices, indptr), shape=(nrows, ncols)).tocsc()

        A = (A_in - A_out).tocsr()

        node_data = self.node_data.reset_index()

        b = np.zeros(A.shape[0])
        b[node_data["node"].values] = node_data["demand"].values

        ub = edge_data["capacity"].values
        obj = edge_data["cost"].values

        varnames = [
            f"flow[{t[0]},{t[1]}]"
            for t in edge_data[["from", "to"]].itertuples(index=False)
        ]
        constrnames = [f"balance[{node}]" for node in range(nrows)]

        with gp.Env() as env, gp.Model(env=env) as model:
            x = model.addMVar(self.edge_data.shape[0], ub=ub, obj=obj, name=varnames)
            model.addMConstr(A, x, "=", b, name=constrnames)
            model.update()
            print(
                f"{model.NumVars=} {model.NumConstrs=} {model.NumNZs=} {model.FingerPrint=}"
            )
            model.write(f"{here}/{self.__class__.__name__}.lp")
