import asyncio
import logging

import aiohttp
import nest_asyncio
import pandas as pd
import pandera.pandas as pa
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

from core_entities.pipelines.drug_llm_tags.drug_atc_codes import get_drug_atc_codes

logger = logging.getLogger(__name__)

nest_asyncio.apply()

# ------------------------------------------------------------
# INGESTION NODES
# ------------------------------------------------------------


def get_boolean_column_schema(column_name: str):
    return pa.Column(
        nullable=False,
        parsers=pa.Parser(lambda col: col.apply(lambda x: True if x == "TRUE" else False)),
        checks=pa.Check(
            lambda col: col.apply(lambda x: x in ["TRUE", "FALSE", True, False]),
            title=f"{column_name} must be True or False",
        ),
    )


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(
            lambda df: df[
                [
                    "id",
                    "name",
                    "synonyms",
                    "aggregated_with",
                    "drug_class",
                    "therapeutic_area",
                    "drug_function",
                    "drug_target",
                    "approved_usa",
                    "is_antipsychotic",
                    "is_sedative",
                    "is_antimicrobial",
                    "is_glucose_regulator",
                    "is_chemotherapy",
                    "is_steroid",
                    "is_analgesic",
                    "is_cardiovascular",
                    "is_cell_therapy",
                    "deleted",
                    "deleted_reason",
                    "new_id",
                ]
            ]
        ),
        columns={
            "id": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.startswith("EC:") and len(x) == 8 and x[3:].isdigit()),
                        title="id does start with 'EC:', is 8 characters long and only contains numbers after EC",
                    ),
                    pa.Check(
                        lambda col: len(col.unique()) == len(col),
                        title="id must be unique",
                    ),
                ],
            ),
            "name": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: "\n" not in x and x.strip() != ""),
                        title="Name must not contain newlines and must not be empty",
                    ),
                    pa.Check(
                        lambda col: len(col.str.lower().unique()) == len(col),
                        title="name must be unique",
                    ),
                ],
            ),
            "synonyms": pa.Column(nullable=True),
            "aggregated_with": pa.Column(nullable=True),
            "drug_class": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: "\n" not in x),
                        title="drug_class must not contain newlines",
                    ),
                ],
            ),
            "therapeutic_area": pa.Column(nullable=True),
            "drug_function": pa.Column(nullable=True),
            "drug_target": pa.Column(nullable=True),
            "approved_usa": pa.Column(
                nullable=False,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: x in ["APPROVED", "NOT_APPROVED", "DISCONTINUED"]),
                    title="approved_usa must be APPROVED, NOT_APPROVED or DISCONTINUED",
                ),
            ),
            "is_antipsychotic": get_boolean_column_schema("is_antipsychotic"),
            "is_sedative": get_boolean_column_schema("is_sedative"),
            "is_antimicrobial": get_boolean_column_schema("is_antimicrobial"),
            "is_glucose_regulator": get_boolean_column_schema("is_glucose_regulator"),
            "is_chemotherapy": get_boolean_column_schema("is_chemotherapy"),
            "is_steroid": get_boolean_column_schema("is_steroid"),
            "is_analgesic": get_boolean_column_schema("is_analgesic"),
            "is_cardiovascular": get_boolean_column_schema("is_cardiovascular"),
            "is_cell_therapy": get_boolean_column_schema("is_cell_therapy"),
            "deleted": pa.Column(
                nullable=True,
                parsers=pa.Parser(lambda col: col.apply(lambda x: True if x == "TRUE" else False)),
            ),
            "deleted_reason": pa.Column(
                nullable=True,
                parsers=pa.Parser(
                    lambda col: col.apply(lambda x: x if isinstance(x, str) and x.strip() != "" else None)
                ),
            ),
            "new_id": pa.Column(
                nullable=True,
                parsers=pa.Parser(
                    lambda col: col.apply(lambda x: x if isinstance(x, str) and x.strip() != "" else None)
                ),
            ),
        },
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x == x.lower()),
                        title="name must be lowercase",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x == x.strip()),
                        title="name must be stripped (no leading/trailing whitespace)",
                    ),
                    pa.Check(
                        lambda col: len(col.str.lower().unique()) == len(col),
                        title="name must be unique",
                    ),
                ],
            ),
            "synonyms": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: isinstance(x, list)),
                        title="synonyms must be a list",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: all(item.lower().strip() == item for item in x)),
                        title="synonyms must contain only stripped lowercase strings",
                    ),
                ],
            ),
            "aggregated_with": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: isinstance(x, list)),
                        title="aggregated_with must be a list",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: all(item.strip().capitalize() == item for item in x)),
                        title="aggregated_with must contain only stripped capitalized strings",
                    ),
                ],
            ),
            "drug_class": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="drug_class must not be empty",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() == x),
                        title="drug_class must not have trailing whitespace",
                    ),
                ],
            ),
            "therapeutic_area": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="therapeutic_area must not be empty",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="therapeutic_area must not have trailing whitespace",
                    ),
                ],
            ),
            "drug_function": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="drug_function must not be empty",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() == x),
                        title="drug_function must not have trailing whitespace",
                    ),
                ],
            ),
            "drug_target": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() == x),
                        title="drug_target must not be empty",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() == x),
                        title="drug_target must not have trailing whitespace",
                    ),
                ],
            ),
            "approved_usa": pa.Column(nullable=False),
            "is_antipsychotic": pa.Column(dtype=bool, nullable=False),
            "is_sedative": pa.Column(dtype=bool, nullable=False),
            "is_antimicrobial": pa.Column(dtype=bool, nullable=False),
            "is_glucose_regulator": pa.Column(dtype=bool, nullable=False),
            "is_chemotherapy": pa.Column(dtype=bool, nullable=False),
            "is_steroid": pa.Column(dtype=bool, nullable=False),
            "is_analgesic": pa.Column(dtype=bool, nullable=False),
            "is_cardiovascular": pa.Column(dtype=bool, nullable=False),
            "is_cell_therapy": pa.Column(dtype=bool, nullable=False),
            "deleted": pa.Column(dtype=bool, nullable=False),
            "deleted_reason": pa.Column(dtype=str, nullable=True),
            "new_id": pa.Column(dtype=str, nullable=True),
        },
        unique=["id"],
        strict=True,
    )
)
def ingest_curated_drug_list(curated_drug_list: pd.DataFrame) -> pd.DataFrame:
    curated_drug_list.loc[:, "name"] = curated_drug_list.loc[:, "name"].apply(lambda x: x.lower().strip())

    def parse_string_column(x: str):
        if x.strip() == "":
            return None
        else:
            return x.strip()

    curated_drug_list.loc[:, "drug_class"] = curated_drug_list.loc[:, "drug_class"].apply(parse_string_column)
    curated_drug_list.loc[:, "therapeutic_area"] = curated_drug_list.loc[:, "therapeutic_area"].apply(
        parse_string_column
    )
    curated_drug_list.loc[:, "drug_function"] = curated_drug_list.loc[:, "drug_function"].apply(parse_string_column)
    curated_drug_list.loc[:, "drug_target"] = curated_drug_list.loc[:, "drug_target"].apply(parse_string_column)

    curated_drug_list.loc[:, "synonyms"] = (
        curated_drug_list["synonyms"]
        .apply(lambda x: [] if pd.isna(x) else [xx.lower().strip() for xx in x.split(";")])
        .apply(lambda x: None if len(x) == 0 else x)
    )
    curated_drug_list.loc[:, "aggregated_with"] = (
        curated_drug_list.loc[:, "aggregated_with"]
        .apply(lambda x: [] if pd.isna(x) else [xx.strip().capitalize() for xx in x.split(";")])
        .apply(lambda x: None if len(x) == 0 else x)
    )

    return curated_drug_list


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(lambda df: df[["drugbank_id", "name", "moldb_smiles"]]),
        columns={
            "drugbank_id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "moldb_smiles": pa.Column(nullable=True),
        },
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "drugbank_id": pa.Column(nullable=False),
            "name": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x == x.lower()),
                        title="name must be lowercase",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x == x.strip()),
                        title="name must be stripped (no leading/trailing whitespace)",
                    ),
                ],
            ),
            "smiles": pa.Column(nullable=True),
        },
        strict=True,
    )
)
def ingest_drugbank_drug_list(drugbank_drug_list: pd.DataFrame) -> pd.DataFrame:
    drugbank_drug_list.loc[:, "name"] = drugbank_drug_list.loc[:, "name"].apply(lambda x: x.lower().strip())
    drugbank_drug_list = drugbank_drug_list.rename(columns={"moldb_smiles": "smiles"})
    return drugbank_drug_list


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(lambda df: df[["drugbank_id", "name", "moldb_smiles"]]),
        columns={
            "drugbank_id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "moldb_smiles": pa.Column(nullable=True),
        },
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "drugbank_id": pa.Column(nullable=False),
            "name": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x == x.lower()),
                        title="name must be lowercase",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x == x.strip()),
                        title="name must be stripped (no leading/trailing whitespace)",
                    ),
                ],
            ),
            "smiles": pa.Column(nullable=True),
        },
        strict=True,
    )
)
def ingest_drugbank_salt_list(drugbank_salt_list: pd.DataFrame) -> pd.DataFrame:
    drugbank_salt_list.loc[:, "name"] = drugbank_salt_list.loc[:, "name"].apply(lambda x: x.lower().strip())
    drugbank_salt_list = drugbank_salt_list.rename(columns={"moldb_smiles": "smiles"})
    return drugbank_salt_list


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(lambda df: df[["Class ID", "Preferred Label"]]),
        columns={
            "Class ID": pa.Column(nullable=False),
            "Preferred Label": pa.Column(nullable=False),
        },
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "atc_code": pa.Column(nullable=False),
            "atc_label": pa.Column(nullable=False),
        },
        strict=True,
    )
)
def ingest_atc_labels(atc: pd.DataFrame) -> pd.DataFrame:
    atc["atc_code"] = atc["Class ID"].apply(
        lambda x: x.replace("http://purl.bioontology.org/ontology/ATC/", "").upper()
    )
    atc["atc_label"] = atc["Preferred Label"].apply(lambda x: x.lower().capitalize())
    atc = atc.drop(columns=["Class ID", "Preferred Label"])
    return atc


# ------------------------------------------------------------
# RESOLUTION NODES
# ------------------------------------------------------------


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(lambda df: df[["id", "name"]]),
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.lower().strip() == x),
                        title="name must be lowercase and stripped",
                    ),
                    pa.Check(
                        lambda col: len(col.str.lower().unique()) == len(col),
                        title="name must be unique",
                    ),
                ],
            ),
        },
        unique=["id"],
        strict=True,
    ),
    obj_getter="curated_drug_list",
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "normalization_success": pa.Column(nullable=False),
            "curie": pa.Column(nullable=False),
            "preferred_label": pa.Column(nullable=False),
            "category": pa.Column(nullable=False),
            "synonyms": pa.Column(nullable=False),
        },
        unique=["id"],
        strict=True,
    ),
    obj_getter="name_resolved_curies",
)
def resolve_drug_curies(
    curated_drug_list: pd.DataFrame,
    name_resolver_base_url: str,
    name_resolver_path: str,
    accepted_drug_categories: list[str],
    parallelism: int,
) -> pd.DataFrame:
    async def get_name_resolver_version(name_resolver_base_url: str) -> str:
        nr_openapi_json_url = f"{name_resolver_base_url}/openapi.json"
        async with aiohttp.ClientSession() as session:
            async with session.get(nr_openapi_json_url) as response:
                json_response = await response.json()
                version = json_response["info"]["version"]
                return version

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    async def resolve_name(id: str, name: str, name_resolver_url: str, session: aiohttp.ClientSession) -> dict:
        try:
            async with session.get(name_resolver_url.format(name=name), timeout=60) as response:
                if response.status != 200:
                    raise Exception(f"Error for name {name}: {response.status}")

                result = await response.json()
                if not result:
                    raise Exception(f"No result for name {name}")

                return {
                    "id": id,
                    "name": name,
                    "normalization_success": True,
                    "output": result,
                }
        except TimeoutError:
            raise Exception(f"Timeout for name: {name}")
        except Exception as e:
            raise Exception(f"Error for name {name}: {str(e)}")

    async def resolve_all_names(
        curated_drug_list: pd.DataFrame,
        name_resolver_url: str,
        parallelism: int,
        max_connections: int,
    ) -> list[dict]:
        semaphore = asyncio.Semaphore(parallelism)

        async def call_name_resolver_with_semaphore(
            id: str, name: str, name_resolver_url: str, session: aiohttp.ClientSession
        ) -> dict:
            async with semaphore:
                return await resolve_name(id, name, name_resolver_url, session)

        # Configure connection pool with increased timeouts
        conn = aiohttp.TCPConnector(limit=max_connections, force_close=True)
        timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_read=60)

        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            tasks = [
                call_name_resolver_with_semaphore(row["id"], row["name"], name_resolver_url, session)
                for _, row in curated_drug_list.iterrows()
            ]
            return await tqdm.gather(*tasks)

    def filter_name_resolver_result(name_resolver_result: dict, accepted_drug_categories: list[str]) -> list[dict]:
        if not name_resolver_result["normalization_success"]:
            logger.warning(f"No normalization success for {name_resolver_result['name']}")
            return {
                "id": name_resolver_result["id"],
                "name": name_resolver_result["name"],
                "normalization_success": False,
                "curie": None,
                "preferred_label": None,
                "category": None,
                "synonyms": None,
            }
        else:
            for match in name_resolver_result.get("output", []):
                if len(set(match["types"]) & set(accepted_drug_categories)) > 0:
                    return {
                        "id": name_resolver_result["id"],
                        "name": name_resolver_result["name"],
                        "normalization_success": True,
                        "curie": match["curie"],
                        "preferred_label": match["label"],
                        "category": [match["types"]],
                        "synonyms": [match["synonyms"]],
                    }
                else:
                    continue

            logger.warning(
                f"No match found for {name_resolver_result['name']} in the accepted_drug_categories. Picking the first match."
            )
            return {
                "id": name_resolver_result["id"],
                "name": name_resolver_result["name"],
                "normalization_success": True,
                "curie": name_resolver_result["output"][0]["curie"],
                "preferred_label": name_resolver_result["output"][0]["label"],
                "category": [name_resolver_result["output"][0]["types"]],
                "synonyms": [name_resolver_result["output"][0]["synonyms"]],
            }

    name_resolver_url = f"{name_resolver_base_url}{name_resolver_path}"
    name_resolver_version = asyncio.run(get_name_resolver_version(name_resolver_base_url))

    # ... call Name Resolver
    name_resolver_results = asyncio.run(
        resolve_all_names(curated_drug_list, name_resolver_url, parallelism, parallelism)
    )
    # ... filter name resolver results
    nameres_df = pd.DataFrame(
        [filter_name_resolver_result(result, accepted_drug_categories) for result in name_resolver_results]
    )

    return {
        "name_resolved_curies": nameres_df,
        "name_resolver_version": name_resolver_version,
    }


@pa.check_input(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "normalization_success": pa.Column(nullable=False),
            "curie": pa.Column(
                nullable=False,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: x.strip() != ""),
                    title="curie must not be empty",
                ),
            ),
            "preferred_label": pa.Column(nullable=False),
            "category": pa.Column(nullable=False),
            "synonyms": pa.Column(nullable=False),
        },
        unique=["id"],
        strict=True,
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "normalization_success": pa.Column(nullable=False),
            "curie": pa.Column(nullable=False),
            "preferred_label": pa.Column(nullable=False),
            "category": pa.Column(nullable=False),
        },
        strict=True,
    ),
    obj_getter="normalized_drug_curies",
)
def normalize_drug_curies(drug_curies: pd.DataFrame, node_normalizer_base_url: str, node_normalizer_path: str) -> dict:
    async def get_nodenorm_version(node_normalizer_base_url: str) -> str:
        nn_openapi_json_url = f"{node_normalizer_base_url}/openapi.json"
        async with aiohttp.ClientSession() as session:
            async with session.get(nn_openapi_json_url) as response:
                json_response = await response.json()
                version = json_response["info"]["version"]
                return version

    async def normalize_curies(curies: list[str], node_normalizer_url: str) -> list[dict]:
        # Configure connection pool with increased timeouts
        conn = aiohttp.TCPConnector(limit=20, force_close=True)
        timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_read=60)

        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            payload = {
                "curies": curies,
                "conflate": True,
                "description": True,
                "drug_chemical_conflate": False,
            }

            results = []
            async with session.post(node_normalizer_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    for curie, node_data in result.items():
                        if node_data:
                            results.append(
                                {
                                    "original_curie": curie,
                                    "normalized_curie": node_data["id"]["identifier"] if node_data.get("id") else None,
                                    "all_categories": node_data.get("type", None),
                                }
                            )
                        else:
                            results.append(
                                {
                                    "original_id": curie,
                                    "id": None,
                                    "all_categories": None,
                                }
                            )

                    return results

                elif response.status == 502:
                    logger.error("Server overloaded (502). Retrying batch after delay...")
                    raise aiohttp.ClientError("Server overloaded")
                elif response.status == 422:
                    raise Exception("Invalid data (422).")
                else:
                    raise Exception(f"Error status {response.status}")

    node_normalizer_url = f"{node_normalizer_base_url}{node_normalizer_path}"
    curies = drug_curies["curie"].unique().tolist()
    nodenorm_version = asyncio.run(get_nodenorm_version(node_normalizer_base_url))
    normalized_results = asyncio.run(normalize_curies(curies, node_normalizer_url))
    normalized_df = pd.DataFrame(data=normalized_results, columns=["original_curie", "normalized_curie"])[
        ["original_curie", "normalized_curie"]
    ]

    failed_normalization = normalized_df[normalized_df["normalized_curie"].isna()]
    if len(failed_normalization) > 0:
        logger.warning(f"Failed to normalize {len(failed_normalization)} drugs")
        logger.warning(failed_normalization.head(15)["original_curie"])
    normalized_df = normalized_df[normalized_df["normalized_curie"].notna()].rename(columns={"original_curie": "curie"})

    merged_df = pd.merge(drug_curies, normalized_df, on="curie", how="left")
    merged_df["curie"] = merged_df["normalized_curie"].combine_first(merged_df["curie"])
    merged_df = merged_df.drop(columns=["normalized_curie", "synonyms"])

    return {"normalized_drug_curies": merged_df, "nodenorm_version": nodenorm_version}


@pa.check_input(
    pa.DataFrameSchema(
        columns={
            "drugbank_id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "smiles": pa.Column(nullable=True),
        },
    ),
    obj_getter="drugbank_drug_list",
)
@pa.check_input(
    pa.DataFrameSchema(
        columns={
            "drugbank_id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "smiles": pa.Column(nullable=True),
        },
    ),
    obj_getter="drugbank_salt_list",
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "drugbank_id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "smiles": pa.Column(nullable=True),
        },
        strict=True,
    )
)
def union_drugbank_lists(drugbank_drug_list: pd.DataFrame, drugbank_salt_list: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([drugbank_drug_list, drugbank_salt_list])


@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "drugbank_id": pa.Column(
                nullable=True,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: pd.notna(x)),
                    title="drugbank_id must not be null",
                    ignore_na=False,
                    raise_warning=True,
                ),
            ),
            "smiles": pa.Column(nullable=True),
        },
        unique=["id"],
        strict=True,
    )
)
def resolve_drugbank_ids(curated_drug_list: pd.DataFrame, drugbank_union_list: pd.DataFrame) -> pd.DataFrame:
    drug_name_and_synonyms = curated_drug_list.explode("synonyms")[["id", "name", "synonyms"]]
    merged_df = pd.merge(
        pd.merge(
            drug_name_and_synonyms,
            drugbank_union_list[["name", "drugbank_id"]].rename(columns={"drugbank_id": "drugbank_id_via_name"}),
            on="name",
            how="left",
        ),
        drugbank_union_list[["name", "drugbank_id"]].rename(columns={"drugbank_id": "drugbank_id_via_synonym"}),
        left_on="synonyms",
        right_on="name",
        how="left",
    )

    merged_df["drugbank_id"] = merged_df["drugbank_id_via_name"].combine_first(merged_df["drugbank_id_via_synonym"])
    merged_df = merged_df.drop(columns=["drugbank_id_via_name", "drugbank_id_via_synonym", "name_y"]).rename(
        columns={"name_x": "name"}
    )
    merged_df = pd.merge(
        merged_df,
        drugbank_union_list[["drugbank_id", "smiles"]],
        on="drugbank_id",
        how="left",
    )
    merged_df = merged_df.drop_duplicates(subset=["name"], keep="first")
    merged_df = merged_df.drop(columns=["name", "synonyms"])

    return merged_df


@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "atc_name": pa.Column(nullable=True),
            "atc_synonym": pa.Column(nullable=True),
            "atc_main": pa.Column(nullable=True),
            "atc_level_1": pa.Column(nullable=True),
            "atc_level_2": pa.Column(nullable=True),
            "atc_level_3": pa.Column(nullable=True),
            "atc_level_4": pa.Column(nullable=True),
            "atc_level_5": pa.Column(nullable=True),
            "l1_label": pa.Column(nullable=True),
            "l2_label": pa.Column(nullable=True),
            "l3_label": pa.Column(nullable=True),
            "l4_label": pa.Column(nullable=True),
            "l5_label": pa.Column(nullable=True),
        },
        strict=True,
    )
)
def resolve_atc_codes(
    curated_drug_list: pd.DataFrame, atc_labels: pd.DataFrame, whocc_parallelism: int = 50
) -> pd.DataFrame:
    async def resolve_all_atc_codes(curated_drug_list: pd.DataFrame, parallelism: int) -> pd.DataFrame:
        semaphore = asyncio.Semaphore(parallelism)

        async def resolve_atc_code_with_semaphore(id: str, name: str, synonyms: list[str]) -> dict:
            async with semaphore:
                atc_codes = await get_drug_atc_codes(name, synonyms, session)
                atc_codes["id"] = id
                return atc_codes

        conn = aiohttp.TCPConnector(limit=parallelism, force_close=True)
        timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_read=60)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            tasks = [
                resolve_atc_code_with_semaphore(row["id"], row["name"], row["synonyms"])
                for _, row in curated_drug_list.iterrows()
            ]
            return await tqdm.gather(*tasks)

    def break_down_atc_code(atc_codes: pd.DataFrame):
        # Level 1: Anatomical main group (first character)
        atc_codes["atc_level_1"] = atc_codes.atc_main.apply(lambda x: x[0] if x is not None and len(x) >= 1 else None)
        # Level 2: Therapeutic subgroup (first 3 characters)
        atc_codes["atc_level_2"] = atc_codes.atc_main.apply(lambda x: x[:3] if x is not None and len(x) >= 3 else None)
        # Level 3: Pharmacological subgroup (first 4 characters)
        atc_codes["atc_level_3"] = atc_codes.atc_main.apply(lambda x: x[:4] if x is not None and len(x) >= 4 else None)
        # Level 4: Chemical subgroup (first 5 characters)
        atc_codes["atc_level_4"] = atc_codes.atc_main.apply(lambda x: x[:5] if x is not None and len(x) >= 5 else None)
        # Level 5: Chemical substance (all 7 characters)
        atc_codes["atc_level_5"] = atc_codes.atc_main.apply(lambda x: x if x is not None and len(x) == 7 else None)

        return atc_codes

    def add_atc_labels(atc_codes: pd.DataFrame, atc_labels: pd.DataFrame):
        return (
            atc_codes.merge(
                atc_labels.rename(columns={"atc_code": "atc_level_1", "atc_label": "l1_label"}),
                how="left",
                on="atc_level_1",
            )
            .merge(
                atc_labels.rename(columns={"atc_code": "atc_level_2", "atc_label": "l2_label"}),
                how="left",
                on="atc_level_2",
            )
            .merge(
                atc_labels.rename(columns={"atc_code": "atc_level_3", "atc_label": "l3_label"}),
                how="left",
                on="atc_level_3",
            )
            .merge(
                atc_labels.rename(columns={"atc_code": "atc_level_4", "atc_label": "l4_label"}),
                how="left",
                on="atc_level_4",
            )
            .merge(
                atc_labels.rename(columns={"atc_code": "atc_level_5", "atc_label": "l5_label"}),
                how="left",
                on="atc_level_5",
            )
        )

    atc_codes = pd.DataFrame(asyncio.run(resolve_all_atc_codes(curated_drug_list, parallelism=whocc_parallelism)))
    atc_codes_broken_down = break_down_atc_code(atc_codes)

    atc_codes_and_labels = add_atc_labels(atc_codes_broken_down, atc_labels)

    result = pd.merge(curated_drug_list[["id", "name"]], atc_codes_and_labels, on="id", how="left")
    return result


def get_log_nan_check(column_name: str):
    return pa.Check(
        lambda col: col.apply(lambda x: pd.notna(x)),
        title=f"{column_name} must not be NaN",
        ignore_na=False,
        raise_warning=True,
    )


@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: len(col.unique()) == len(col),
                        title="id must be unique",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x.startswith("EC:") and len(x) == 8 and x[3:].isdigit()),
                        title="id does start with 'EC:', is 8 characters long and only contains numbers after EC",
                    ),
                ],
            ),
            "name": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="name must not be empty",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x[0].isupper()),
                        title="name must start with a capital letter",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x == x.strip()),
                        title="name must be stripped (no leading/trailing whitespace)",
                    ),
                ],
            ),
            "translator_id": pa.Column(
                nullable=True,
                checks=pa.Check(
                    lambda col: len(col.unique()) == len(col),
                    title="translator_id must be unique",
                ),
            ),
            "drugbank_id": pa.Column(
                nullable=True,
                checks=[
                    get_log_nan_check("drugbank_id"),
                    pa.Check(
                        lambda col: len(col[col.notna()].unique()) == len(col[col.notna()]),
                        title="drugbank_id must not have duplicates",
                        raise_warning=True,
                    ),
                ],
            ),
            "synonyms": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: all(item.capitalize().strip() == item for item in x)),
                        title="aggregated_with must contain only stripped capitalized strings",
                    ),
                ],
            ),
            "aggregated_with": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: all(item.capitalize().strip() == item for item in x)),
                        title="aggregated_with must contain only stripped capitalized strings",
                    ),
                ],
            ),
            "drug_class": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="drug_class must not be empty",
                    ),
                ],
            ),
            "therapeutic_area": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="therapeutic_area must not be empty",
                    ),
                ],
            ),
            "drug_function": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="drug_function must not be empty",
                    ),
                ],
            ),
            "drug_target": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="drug_target must not be empty",
                    ),
                ],
            ),
            "approved_usa": pa.Column(nullable=False),
            "is_antipsychotic": pa.Column(dtype=bool, nullable=False),
            "is_sedative": pa.Column(dtype=bool, nullable=False),
            "is_antimicrobial": pa.Column(dtype=bool, nullable=False),
            "is_glucose_regulator": pa.Column(dtype=bool, nullable=False),
            "is_chemotherapy": pa.Column(dtype=bool, nullable=False),
            "is_steroid": pa.Column(dtype=bool, nullable=False),
            "is_analgesic": pa.Column(dtype=bool, nullable=False),
            "is_cardiovascular": pa.Column(dtype=bool, nullable=False),
            "is_cell_therapy": pa.Column(dtype=bool, nullable=False),
            "smiles": pa.Column(nullable=True),
            "atc_main": pa.Column(nullable=True),
            "atc_level_1": pa.Column(nullable=True),
            "atc_level_2": pa.Column(nullable=True),
            "atc_level_3": pa.Column(nullable=True),
            "atc_level_4": pa.Column(nullable=True),
            "atc_level_5": pa.Column(nullable=True),
            "l1_label": pa.Column(nullable=True),
            "l2_label": pa.Column(nullable=True),
            "l3_label": pa.Column(nullable=True),
            "l4_label": pa.Column(nullable=True),
            "l5_label": pa.Column(nullable=True),
            "deleted": pa.Column(dtype=bool, nullable=False),
            "deleted_reason": pa.Column(dtype=str, nullable=True),
            "new_id": pa.Column(dtype=str, nullable=True),
        },
        unique=["id"],
        strict=True,
    ),
    obj_getter=0,
)
def merge_drug_lists(
    curated_drug_list: pd.DataFrame,
    normalized_drug_curies: pd.DataFrame,
    drug_list_with_atc_codes: pd.DataFrame,
    drug_list_with_drugbank_id: pd.DataFrame,
    release_columns: list[str],
    drug_exception_list: list[str],
) -> pd.DataFrame:
    normalized_drug_curies = normalized_drug_curies.rename(columns={"curie": "translator_id"})

    df = (
        curated_drug_list.merge(normalized_drug_curies, on="id", how="left")
        .merge(drug_list_with_atc_codes, on="id", how="left")
        .merge(drug_list_with_drugbank_id, on="id", how="left")
    )

    df = df[~df["name"].isin(drug_exception_list)]
    df.loc[:, "name"] = df.loc[:, "name"].apply(lambda x: x.capitalize())
    df.loc[:, "synonyms"] = df.loc[:, "synonyms"].apply(
        lambda x: None if x is None else [xx.strip().capitalize() for xx in x]
    )
    df = df[release_columns]

    return df, df


def publish_drug_list(drug_list: pd.DataFrame) -> dict:
    return {
        "drug_list_parquet": drug_list,
        "drug_list_tsv": drug_list,
        "drug_list_bq": drug_list,
        "drug_list_bq_latest": drug_list,
    }
