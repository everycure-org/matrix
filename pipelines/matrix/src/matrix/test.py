# %%

import mlflow

mlflow.set_tracking_uri("http://localhost:5001")

exp = mlflow.search_experiments(filter_string="name = 'default'")
if len(exp) == 1:
    print(exp[0].experiment_id)


# %%
