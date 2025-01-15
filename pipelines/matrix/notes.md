# TODO delete or move before mergin


- disable any KG except RTX in settings.py
- Run full pipeline in `-e sample` to prep all the other datasets
    - `kedro run -e sample --to-nodes create_prm_unified_edges,create_prm_unified_nodes`
- run `kedro run -p sample -e sample` to overwrite rtx_kg2 with a sample only
