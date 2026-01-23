import logging

import pandas as pd
import pandera.pandas as pa

from core_entities.utils.curation_utils import _log_merge_statistics, apply_patch

logger = logging.getLogger(__name__)


def _get_curated_list_boolean_check(column_name: str):
    return pa.Check(
        lambda col: col.apply(lambda x: x in ["True", "False"] or x.strip() == ""),
        title=f"{column_name} value is valid",
    )


curated_list_speciality_columns = [
    "speciality_breast",
    "speciality_cardiovascular",
    "speciality_chromosomal",
    "speciality_connective_tissue",
    "speciality_dermatologic",
    "speciality_ear_nose_throat",
    "speciality_endocrine",
    "speciality_eye_and_adnexa",
    "speciality_gastrointestinal",
    "speciality_hematologic",
    "speciality_immune",
    "speciality_infection",
    "speciality_metabolic",
    "speciality_musculoskeletal",
    "speciality_neoplasm",
    "speciality_neurological",
    "speciality_obstetric",
    "speciality_poisoning_and_toxicity",
    "speciality_psychiatric",
    "speciality_reproductive",
    "speciality_respiratory",
    "speciality_renal_and_urinary",
    "speciality_syndromic",
]


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(
            lambda df: df[
                [
                    "category_class",
                    "label",
                    "synonyms",
                    "harrisons_view",
                    "mondo_txgnn",
                    "mondo_top_grouping",
                ]
            ]
        ),
        columns={
            "category_class": pa.Column(nullable=False),
            "label": pa.Column(nullable=False),
            "synonyms": pa.Column(nullable=True),
            "harrisons_view": pa.Column(nullable=True),
            "mondo_txgnn": pa.Column(nullable=True),
            "mondo_top_grouping": pa.Column(nullable=True),
        },
        unique=["category_class"],
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "id": pa.Column(dtype=str, nullable=False),
            "name": pa.Column(dtype=str, nullable=False),
            "synonyms": pa.Column(dtype=str, nullable=True),
            "harrisons_view": pa.Column(dtype=str, nullable=True),
            "mondo_txgnn": pa.Column(dtype=str, nullable=False),
            "mondo_top_grouping": pa.Column(dtype=str, nullable=True),
        },
        strict=True,
        unique=["id"],
    )
)
def ingest_disease_list(disease_list: pd.DataFrame) -> pd.DataFrame:
    disease_list = disease_list.rename(columns={"category_class": "id", "label": "name"})

    # Change name to start with an uppercase letter
    disease_list["name"] = disease_list["name"].str.capitalize()

    # Fill null values
    def parse_string_column(x: str):
        if pd.isna(x) or x.strip() == "":
            return None
        else:
            return x

    disease_list["synonyms"] = disease_list["synonyms"].apply(parse_string_column)
    disease_list["harrisons_view"] = disease_list["harrisons_view"].apply(parse_string_column)
    disease_list["mondo_txgnn"] = disease_list["mondo_txgnn"].fillna("other")
    disease_list["mondo_top_grouping"] = disease_list["mondo_top_grouping"].apply(parse_string_column)

    return disease_list


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(
            lambda df: df[
                [
                    "mondo_id",
                    "level",
                    "supergroup",
                    *curated_list_speciality_columns,
                    "core",
                    "anatomical_deformity",
                    "benign_malignant",
                    "precancerous",
                    "anatomical_id",
                    "anatomical_name",
                ]
            ]
        ),
        columns={
            "mondo_id": pa.Column(
                nullable=False,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: x.startswith("MONDO:")),
                    title="mondo_id does not start with 'MONDO:'",
                ),
            ),
            "level": pa.Column(
                nullable=True,
                checks=pa.Check(
                    lambda col: col.apply(
                        lambda x: x.strip() == "" or x in ["clinically_recognized", "subgroup", "exclude", "grouping"]
                    ),
                    ignore_na=False,
                    title="level value is valid",
                ),
            ),
            "supergroup": pa.Column(
                nullable=True,
                checks=pa.Check(
                    lambda col: col.isin(["NNNI", "neoplasm", "infection", "exclude"]) | col.isna(),
                    ignore_na=False,
                    title="supergroup value is valid",
                ),
            ),
            **{
                col: pa.Column(nullable=True, checks=_get_curated_list_boolean_check(col))
                for col in curated_list_speciality_columns
            },
            "core": pa.Column(nullable=True, checks=_get_curated_list_boolean_check("core")),
            "anatomical_deformity": pa.Column(
                nullable=True,
                checks=_get_curated_list_boolean_check("anatomical_deformity"),
            ),
            "benign_malignant": pa.Column(
                nullable=True,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: x.strip() == "" or x in ["benign", "malignant"]),
                    ignore_na=False,
                    title="benign_malignant value is valid",
                ),
            ),
            "precancerous": pa.Column(nullable=True, checks=_get_curated_list_boolean_check("precancerous")),
            "anatomical_id": pa.Column(
                nullable=True,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: x.strip() == "" or x == "null" or x.startswith("MONDO:")),
                    title="anatomical_id does not start with 'MONDO:'",
                ),
            ),
            "anatomical_name": pa.Column(nullable=True),
        },
        unique=["mondo_id"],
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "mondo_id": pa.Column(dtype=str, nullable=False),
            "level": pa.Column(dtype=str, nullable=True),
            "supergroup": pa.Column(dtype=str, nullable=True),
            **{col: pa.Column(dtype=bool, nullable=True) for col in curated_list_speciality_columns},
            "core": pa.Column(dtype=bool, nullable=True),
            "anatomical_deformity": pa.Column(dtype=bool, nullable=True),
            "benign_malignant": pa.Column(dtype=str, nullable=True),
            "precancerous": pa.Column(dtype=bool, nullable=True),
            "anatomical_id": pa.Column(dtype=str, nullable=True),
            "anatomical_name": pa.Column(dtype=str, nullable=True),
        },
        unique=["mondo_id"],
    )
)
def ingest_curated_disease_list(curated_disease_list: pd.DataFrame) -> pd.DataFrame:
    def parse_string_column(x: str):
        if x.strip() == "" or pd.isna(x) or x == "null":
            return None
        else:
            return x.replace("\n", "")

    curated_disease_list.loc[:, "level"] = curated_disease_list["level"].apply(parse_string_column)
    curated_disease_list.loc[:, "supergroup"] = curated_disease_list["supergroup"].apply(parse_string_column)
    curated_disease_list.loc[:, "benign_malignant"] = curated_disease_list["benign_malignant"].apply(
        parse_string_column
    )
    curated_disease_list.loc[:, "anatomical_id"] = curated_disease_list["anatomical_id"].apply(parse_string_column)
    curated_disease_list.loc[:, "anatomical_name"] = curated_disease_list["anatomical_name"].apply(parse_string_column)

    def parse_boolean_column(x: str):
        if x == "True":
            return True
        elif x == "False":
            return False
        else:
            return None

    for col in curated_list_speciality_columns + [
        "core",
        "anatomical_deformity",
        "precancerous",
    ]:
        curated_disease_list.loc[:, col] = curated_disease_list[col].apply(parse_boolean_column)

    dtypes_dict = {
        **{col: bool for col in curated_list_speciality_columns + ["core", "anatomical_deformity", "precancerous"]},
    }

    curated_disease_list = curated_disease_list.astype(dtypes_dict)

    return curated_disease_list


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(lambda df: df[["id", "name"]]),
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
        },
        strict=True,
        unique=["id"],
    )
)
def ingest_disease_name_patch(disease_name_patch: pd.DataFrame) -> pd.DataFrame:
    return disease_name_patch


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(
            lambda df: df[
                [
                    "id",
                    "is_psychiatric_disease",
                    "is_malignant_cancer",
                    "is_benign_tumour",
                    "is_pathogen_caused",
                    "is_glucose_dysfunction",
                ]
            ]
        ),
        columns={
            "id": pa.Column(nullable=False),
            "is_psychiatric_disease": pa.Column(nullable=False),
            "is_malignant_cancer": pa.Column(nullable=False),
            "is_benign_tumour": pa.Column(nullable=False),
            "is_pathogen_caused": pa.Column(nullable=False),
            "is_glucose_dysfunction": pa.Column(nullable=False),
        },
        unique=["id"],
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "id": pa.Column(dtype=str, nullable=False),
            "is_psychiatric_disease": pa.Column(nullable=False),
            "is_malignant_cancer": pa.Column(nullable=False),
            "is_benign_tumour": pa.Column(nullable=False),
            "is_pathogen_caused": pa.Column(nullable=False),
            "is_glucose_dysfunction": pa.Column(nullable=False),
        },
        strict=True,
        unique=["id"],
    )
)
def ingest_disease_categories(disease_categories: pd.DataFrame) -> pd.DataFrame:
    return disease_categories


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(
            lambda df: df[
                [
                    "id",
                    "umn_score",
                ]
            ]
        ),
        columns={
            "id": pa.Column(nullable=False),
            "umn_score": pa.Column(nullable=False),
        },
        strict=True,
        unique=["id"],
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "id": pa.Column(dtype=str, nullable=False),
            "unmet_medical_need": pa.Column(dtype=float, nullable=False),
        },
        strict=True,
        unique=["id"],
    )
)
def ingest_disease_umn(
    disease_umn: pd.DataFrame,
) -> pd.DataFrame:
    return disease_umn.rename(columns={"umn_score": "unmet_medical_need"})


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(
            lambda df: df[
                [
                    "id",
                    "primary_disease_world_prevalence_explanation",
                    "experimental",
                    "disease_world_prevalence",
                ]
            ]
        ),
        columns={
            "id": pa.Column(nullable=False),
            "primary_disease_world_prevalence_explanation": pa.Column(nullable=False),
            "experimental": pa.Column(nullable=False),
            "disease_world_prevalence": pa.Column(nullable=False),
        },
        unique=["id"],
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "id": pa.Column(dtype=str, nullable=False),
            "prevalence_experimental": pa.Column(
                nullable=True,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: x is None or x in ["True", "False"]),
                    title="prevalence_experimental is valid",
                ),
            ),
            "prevalence_world": pa.Column(
                dtype=str,
                nullable=False,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: x.strip() != "" and x.strip() != "null"),
                    title="prevalence_world is not empty",
                ),
            ),
        },
        strict=True,
        unique=["id"],
    )
)
def ingest_disease_prevalence(disease_prevalence: pd.DataFrame) -> pd.DataFrame:
    disease_prevalence.loc[:, "prevalence_experimental"] = disease_prevalence["experimental"].apply(
        lambda x: x if x in ["True", "False"] else None
    )
    disease_prevalence.loc[:, "prevalence_world"] = disease_prevalence.apply(
        lambda row: "prevalence not generated because disease is iatrogenic or drug-induced"
        if row["primary_disease_world_prevalence_explanation"]
        == "['prevalence not generated because disease is iatrogenic or drug-induced']"
        else row["disease_world_prevalence"],
        axis=1,
    )

    return disease_prevalence.drop(
        columns=[
            "primary_disease_world_prevalence_explanation",
            "experimental",
            "disease_world_prevalence",
        ]
    )


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(
            lambda df: df[
                [
                    "id",
                    "txgnn",
                ]
            ]
        ),
        columns={
            "id": pa.Column(nullable=False),
            "txgnn": pa.Column(nullable=False),
        },
        strict=True,
        unique=["id"],
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "id": pa.Column(dtype=str, nullable=False),
            "txgnn": pa.Column(dtype=str, nullable=False),
        },
        strict=True,
        unique=["id"],
    )
)
def ingest_disease_txgnn(disease_txgnn: pd.DataFrame) -> pd.DataFrame:
    return disease_txgnn


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(
            lambda df: df[
                [
                    "id",
                    "disease_kg_node_id",
                    "comment",
                    "source",
                    "status",
                    "archetype",
                    "created_at",
                ]
            ]
        ),
        columns={
            "id": pa.Column(nullable=False),
            "disease_kg_node_id": pa.Column(nullable=False),
            "comment": pa.Column(nullable=True),
            "source": pa.Column(nullable=True),
            "status": pa.Column(nullable=True),
            "archetype": pa.Column(nullable=True),
            "created_at": pa.Column(nullable=True),
        },
        strict=True,
    )
)
def ingest_orchard_reviews(orchard_reviews: pd.DataFrame) -> pd.DataFrame:
    return orchard_reviews.rename(columns={"id": "review_id"})


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(lambda df: df[["cls", "label", "deprecated", "replacements"]]),
        columns={
            "cls": pa.Column(nullable=False),
            "label": pa.Column(nullable=False),
            "deprecated": pa.Column(nullable=True),
            "replacements": pa.Column(nullable=True),
        },
        strict=True,
        unique=["cls"],
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "id": pa.Column(dtype=str, nullable=False),
            "name": pa.Column(dtype=str, nullable=False),
            "deprecated": pa.Column(dtype=bool, nullable=True),
            "replacements": pa.Column(dtype=str, nullable=True),
        },
        unique=["id"],
    )
)
def ingest_disease_obsolete(disease_obsolete: pd.DataFrame) -> pd.DataFrame:
    disease_obsolete = disease_obsolete.rename(columns={"cls": "id", "label": "name"})
    disease_obsolete.loc[:, "deprecated"] = disease_obsolete["deprecated"].astype(bool)
    disease_obsolete.loc[:, "replacements"] = (
        disease_obsolete["replacements"].apply(lambda x: x if x.strip() != "" else None).astype(str)
    )
    return disease_obsolete


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(lambda df: df[["id", "name", "new_id"]]),
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "new_id": pa.Column(nullable=True),
        },
        strict=True,
        unique=["id"],
    )
)
def ingest_manual_disease_remapping(
    manual_disease_remapping: pd.DataFrame,
) -> pd.DataFrame:
    manual_disease_remapping["deleted"] = True
    manual_disease_remapping.loc[:, "new_id"] = (
        manual_disease_remapping["new_id"].apply(lambda x: x if x.strip() != "" else None).astype(str)
    )
    return manual_disease_remapping


@pa.check_output(
    pa.DataFrameSchema(
        {
            "id": pa.Column(dtype=str, nullable=False),
            "level": pa.Column(dtype=str, nullable=True),
            "supergroup": pa.Column(dtype=str, nullable=True),
            **{col: pa.Column(dtype=bool, nullable=True) for col in curated_list_speciality_columns},
            "core": pa.Column(dtype=bool, nullable=True),
            "anatomical_deformity": pa.Column(dtype=bool, nullable=True),
            "benign_malignant": pa.Column(dtype=str, nullable=True),
            "precancerous": pa.Column(dtype=bool, nullable=True),
            "anatomical_id": pa.Column(dtype=str, nullable=True),
            "anatomical_name": pa.Column(dtype=str, nullable=True),
            "unmet_medical_need": pa.Column(dtype=float, nullable=True),
            "prevalence_experimental": pa.Column(nullable=True),
            "prevalence_world": pa.Column(dtype=str, nullable=True),
        },
        unique=["id"],
    )
)
def merge_disease_lists(
    disease_list: pd.DataFrame,
    disease_categories: pd.DataFrame,
    disease_umn: pd.DataFrame,
    disease_prevalence: pd.DataFrame,
    disease_txgnn: pd.DataFrame,
    curated_disease_list: pd.DataFrame,
) -> pd.DataFrame:
    _log_merge_statistics(
        primary_df=disease_list,
        secondary_df=disease_categories,
        primary_name="disease list",
        secondary_name="disease categories",
        primary_only_action="will be kept with nulls",
        secondary_only_action="will be dropped",
    )
    disease_list = pd.merge(disease_list, disease_categories, on="id", how="left")

    _log_merge_statistics(
        primary_df=disease_list,
        secondary_df=disease_umn,
        primary_name="disease list",
        secondary_name="disease umn",
        primary_only_action="will be kept with nulls",
        secondary_only_action="will be dropped",
    )
    disease_list = pd.merge(
        disease_list,
        disease_umn,
        on="id",
        how="left",
    )

    _log_merge_statistics(
        primary_df=disease_list,
        secondary_df=disease_prevalence,
        primary_name="disease list",
        secondary_name="disease prevalence",
        primary_only_action="will be kept with nulls",
        secondary_only_action="will be dropped",
    )
    disease_list = pd.merge(
        disease_list,
        disease_prevalence,
        on="id",
        how="left",
    )

    _log_merge_statistics(
        primary_df=disease_list,
        secondary_df=disease_txgnn,
        primary_name="disease list",
        secondary_name="disease txgnn",
        primary_only_action="will be kept with nulls",
        secondary_only_action="will be dropped",
    )
    disease_list = pd.merge(
        disease_list,
        disease_txgnn,
        on="id",
        how="left",
    )

    curated_disease_list_renamed = curated_disease_list.rename(columns={"mondo_id": "id"})
    _log_merge_statistics(
        primary_df=disease_list,
        secondary_df=curated_disease_list_renamed,
        primary_name="disease list",
        secondary_name="curated list",
        primary_only_action="will be dropped",
        secondary_only_action="will be dropped",
    )
    merged_disease_list = pd.merge(disease_list, curated_disease_list_renamed, on="id", how="inner")

    if merged_disease_list.isna().any().any():
        logger.warning("⚠️ Disease list has null values")
        null_counts = merged_disease_list.isna().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        logger.warning(f"Columns with null values:\n{columns_with_nulls}")

        if merged_disease_list[merged_disease_list.level == "clinically_recognized"].isna().any().any():
            logger.warning("Out of which, Clinically recognized diseases have null values:")
            null_counts = merged_disease_list[merged_disease_list.level == "clinically_recognized"].isna().sum()
            columns_with_nulls = null_counts[null_counts > 0]
            logger.warning(f"CRD columns with null values:\n{columns_with_nulls}")

    return merged_disease_list


@pa.check_output(
    pa.DataFrameSchema(
        {
            "id": pa.Column(dtype=str, nullable=False),
            "name": pa.Column(dtype=str, nullable=True),
        },
        unique=["id"],
    )
)
def apply_name_patch(disease_list: pd.DataFrame, disease_name_patch: pd.DataFrame) -> pd.DataFrame:
    disease_list = apply_patch(disease_list, disease_name_patch, ["name"], merge_on="id")
    return disease_list


@pa.check_output(
    pa.DataFrameSchema(
        {
            "id": pa.Column(dtype=str, nullable=False),
            "name": pa.Column(
                dtype=str,
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(
                            lambda name: True if not name[0].isalpha() or name[0].isupper() else False
                        ),
                        title="Name must start with a capital letter",
                    )
                ],
            ),
            "synonyms": pa.Column(dtype=str, nullable=True),
            "level": pa.Column(
                nullable=True,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: x in ["clinically_recognized", "subgroup", "exclude", "grouping"]),
                    title="level value is valid",
                ),
            ),
            "supergroup": pa.Column(dtype=str, nullable=False),
            **{col: pa.Column(dtype=bool, nullable=False) for col in curated_list_speciality_columns},
            "core": pa.Column(dtype=bool, nullable=False),
            "anatomical_deformity": pa.Column(dtype=bool, nullable=False),
            "benign_malignant": pa.Column(dtype=str, nullable=True),
            "precancerous": pa.Column(dtype=bool, nullable=True),
            "anatomical_id": pa.Column(dtype=str, nullable=True),
            "anatomical_name": pa.Column(dtype=str, nullable=True),
            "unmet_medical_need": pa.Column(nullable=True),
            "prevalence_experimental": pa.Column(nullable=True),
            "prevalence_world": pa.Column(nullable=True),
            "is_psychiatric_disease": pa.Column(nullable=True),
            "is_malignant_cancer": pa.Column(nullable=True),
            "is_benign_tumour": pa.Column(nullable=True),
            "is_infectious_disease": pa.Column(nullable=True),
            "txgnn": pa.Column(dtype=str, nullable=True),
            "is_glucose_dysfunction": pa.Column(nullable=True),
            "harrisons_view": pa.Column(dtype=str, nullable=True),
            "mondo_txgnn": pa.Column(dtype=str, nullable=False),
            "mondo_top_grouping": pa.Column(dtype=str, nullable=True),
        },
        strict=True,
        unique=["id"],
        checks=[
            pa.Check(
                lambda df: (df != "None").all().all(),
                title="Should not contain None as a string",
            )
        ],
    )
)
# TODO: add value checks for columns that apply for it(level, supergroup ...)
def format_disease_list(disease_list: pd.DataFrame, release_columns: list[str]) -> pd.DataFrame:
    disease_list_sorted = disease_list.sort_values(by="name").reset_index(drop=True)

    disease_list_sorted["is_infectious_disease"] = disease_list_sorted["is_pathogen_caused"] | disease_list_sorted[
        "name"
    ].str.contains(
        "infection|bacterial|viral|virus|fungal|abscess|pneumoni|cocc|staph|strep|gonorr|parasit|tubercul|polio|candid|toxoplasm|syphilis|latent|herpe|fever|mycobact|pox$|poxvirus",
        case=False,
    )
    disease_list_sorted = disease_list_sorted.drop(columns=["is_pathogen_caused"])
    return disease_list_sorted[release_columns]


@pa.check_output(
    pa.DataFrameSchema(
        {
            "id": pa.Column(dtype=str, nullable=False),
            "name": pa.Column(nullable=False),
            "synonyms": pa.Column(nullable=True),
            "level": pa.Column(nullable=True),
            "supergroup": pa.Column(nullable=True),
            **{col: pa.Column(nullable=True) for col in curated_list_speciality_columns},
            "core": pa.Column(nullable=True),
            "anatomical_deformity": pa.Column(nullable=True),
            "benign_malignant": pa.Column(nullable=True),
            "precancerous": pa.Column(nullable=True),
            "anatomical_id": pa.Column(nullable=True),
            "anatomical_name": pa.Column(nullable=True),
            "unmet_medical_need": pa.Column(nullable=True),
            "prevalence_experimental": pa.Column(nullable=True),
            "prevalence_world": pa.Column(nullable=True),
            "is_psychiatric_disease": pa.Column(nullable=True),
            "is_malignant_cancer": pa.Column(nullable=True),
            "is_benign_tumour": pa.Column(nullable=True),
            "harrisons_view": pa.Column(nullable=True),
            "is_infectious_disease": pa.Column(nullable=True),
            "txgnn": pa.Column(nullable=True),
            "mondo_txgnn": pa.Column(nullable=True),
            "mondo_top_grouping": pa.Column(nullable=True),
            "is_glucose_dysfunction": pa.Column(nullable=True),
            "new_id": pa.Column(nullable=True),
            "deleted": pa.Column(nullable=True),
        },
        strict=True,
        unique=["id"],
    ),
    obj_getter="disease_list_with_migrations_parquet",
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "review_id": pa.Column(dtype=int, nullable=False),
            "column": pa.Column(dtype=str, nullable=False),
            "old_value": pa.Column(dtype=str, nullable=False),
            "new_value": pa.Column(dtype=str, nullable=True),
        },
        unique=["review_id"],
    ),
    obj_getter="reviews_to_map",
)
def migrate_diseases_with_dangling_reviews(
    disease_list: pd.DataFrame,
    disease_obsolete: pd.DataFrame,
    orchard_reviews: pd.DataFrame,
    manual_disease_remapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    You can find diagrame illustrating the migration steps in the docs/disease_migration folder
    """
    # 1. Get dangling reviews because of missing entry in disease list (done)
    dangling_reviews = pd.merge(
        orchard_reviews,
        disease_list,
        left_on="disease_kg_node_id",
        right_on="id",
        how="left",
    )
    dangling_reviews = dangling_reviews[dangling_reviews["id"].isna()]
    diseases_with_dangling_reviews = dangling_reviews["disease_kg_node_id"].drop_duplicates()

    # 2. Create remapping using obsolete dataset
    remapping_with_obsolete = pd.merge(
        diseases_with_dangling_reviews,
        disease_obsolete,
        left_on="disease_kg_node_id",
        right_on="id",
        how="left",
    )
    remapping_with_obsolete = remapping_with_obsolete[(remapping_with_obsolete["replacements"].notna())]
    remapping_with_obsolete_df = remapping_with_obsolete.rename(
        columns={"replacements": "new_id", "deprecated": "deleted"}
    )[["id", "name", "new_id", "deleted"]]

    # 3. Add mappings to disease list to avoid dangling reviews
    disease_list["new_id"] = None
    disease_list["deleted"] = False

    manual_disease_remapping[manual_disease_remapping == "None"] = None
    remapping_with_obsolete_df[remapping_with_obsolete_df == "None"] = None

    disease_list_with_migrations = pd.concat(
        [disease_list, remapping_with_obsolete_df, manual_disease_remapping]
    ).reset_index(drop=True)

    # 4. Test that no reviews are dangling with the output list
    dangling_reviews = pd.merge(
        orchard_reviews,
        disease_list_with_migrations,
        left_on="disease_kg_node_id",
        right_on="id",
        how="left",
    )

    disease_id_used_for_other_reviews = "MONDO:0000000"
    dangling_reviews = dangling_reviews[
        (dangling_reviews["id"].isna()) & (dangling_reviews["disease_kg_node_id"] != disease_id_used_for_other_reviews)
    ]
    if len(dangling_reviews) > 0:
        logger.warning(f"⚠️ There are {len(dangling_reviews)} dangling reviews after the remapping")
        logger.warning(dangling_reviews["disease_kg_node_id"])
        raise ValueError("There are dangling reviews with the output list")

    # 5. Provide mapped reviews for Orchard
    obsolete_diseases = pd.concat([remapping_with_obsolete_df, manual_disease_remapping])
    reviews_to_map = pd.merge(
        orchard_reviews,
        obsolete_diseases,
        left_on="disease_kg_node_id",
        right_on="id",
        how="inner",
    )
    reviews_to_map = reviews_to_map.rename(columns={"disease_kg_node_id": "old_value", "new_id": "new_value"})
    reviews_to_map["column"] = "disease_kg_node_id"
    reviews_to_map = reviews_to_map[["review_id", "column", "old_value", "new_value"]]
    reviews_to_map = reviews_to_map[reviews_to_map["new_value"] != "None"]

    return {
        "disease_list_with_migrations_parquet": disease_list_with_migrations,
        "disease_list_with_migrations_tsv": disease_list_with_migrations,
        "reviews_to_map": reviews_to_map,
    }


def publish_disease_list(disease_list: pd.DataFrame) -> dict:
    return {
        "disease_list_parquet": disease_list,
        "disease_list_tsv": disease_list,
        "disease_list_bq": disease_list,
        "disease_list_bq_latest": disease_list,
    }
