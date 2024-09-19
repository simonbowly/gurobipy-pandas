
Clone repo:

```
databricks repos create \
    https://github.com/Gurobi/gurobipy-pandas.git \
    --path /Shared/gurobipy-pandas
```

Set up job:

```
databricks jobs create --json @workflow.json
```
