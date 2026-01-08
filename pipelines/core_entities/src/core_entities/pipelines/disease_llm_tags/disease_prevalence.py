import logging
from collections import Counter
from operator import add
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from PIL import Image
from pydantic import BaseModel

from core_entities.utils.llm_utils import InvokableGraph, LLMConfig, get_llm_response

logger = logging.getLogger(__name__)


class PrimaryPrevalenceResult(BaseModel):
    disease_world_prevalence_explanation: str
    disease_world_prevalence: str
    unsure: bool


class SubtypePrevalenceResult(BaseModel):
    parent_disease: str
    parent_disease_world_prevalence_explanation: str
    parent_disease_world_prevalence: str
    subtype_percentage_occurrence_explanation: str
    subtype_percentage_occurrence: float
    unsure: bool


class PopulationSubgroupResult(BaseModel):
    subgroup_specified_explanation: str
    subgroup_specified: bool
    population_subgroup: str
    population_subgroup_size: float


class ExtractPrevalenceData(BaseModel):
    categorisation_explanation: str
    category: Literal["percentage", "ratio", "number"]
    numerator: float
    denominator: float


class DiseasePrevalenceGraph(InvokableGraph):
    class State(TypedDict):
        # input
        disease_name: str
        synonyms: str
        synonyms_prompt: str
        disease_label: str

        # assets
        primary_disease_world_prevalence_explanation: str
        primary_disease_world_prevalence: str
        subtype_parent_disease: str
        subtype_percentage_occurrence_explanation: str
        subtype_percentage_occurrence: float
        unsure: bool
        subgroup_specified_explanation: str
        subgroup_specified: bool
        population_subgroup: str
        population_subgroup_size: float
        prevalence_data_explanation: str
        prevalence_data_type: str
        prevalence_data_numerator: float
        prevalence_data_denominator: float
        prevalence_in_100k: float

        # final response
        disease_world_prevalence_subgraph: str

        # token counter
        request_token_counter: Annotated[list[tuple[int, str]], add]
        response_token_counter: Annotated[list[tuple[int, str]], add]

    def __init__(self, config: dict):
        self.config = config
        self._setup_config_attributes()

        super().__init__()

    def _setup_config_attributes(self):
        self.primary_prevalence = LLMConfig(self.config.get("primary_prevalence"))
        self.subtype_prevalence = LLMConfig(self.config.get("subtype_prevalence"))
        self.population_subgroup = LLMConfig(self.config.get("population_subgroup"))
        self.extract_prevalence = LLMConfig(self.config.get("extract_prevalence_data"))

    def initialise_state(self, state: State) -> State:
        return {
            "request_token_counter": [],
            "response_token_counter": [],
            "synonyms_prompt": f"(This entity is also known as {state['synonyms']})" if state["synonyms"] else "",
        }

    def check_disease_label(self, state: State) -> str:
        if state["disease_label"] == "disease-primary" or not state["disease_label"]:
            return "disease_is_primary"
        elif state["disease_label"] == "disease-subtype":
            return "disease_is_subtype"

    async def get_primary_prevalence(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.primary_prevalence.prompt.format(
                disease_name=state["disease_name"],
                synonyms_prompt=state["synonyms_prompt"],
            ),
            model_config=self.primary_prevalence.model_config,
            pydantic_model=PrimaryPrevalenceResult,
            system_prompt=self.primary_prevalence.system_prompt,
        )

        return {
            "primary_disease_world_prevalence_explanation": response.output.disease_world_prevalence_explanation,
            "primary_disease_world_prevalence": response.output.disease_world_prevalence,
            "unsure": response.output.unsure,
            "request_token_counter": [("primary_disease_world_prevalence", response.usage().request_tokens)],
            "response_token_counter": [("primary_disease_world_prevalence", response.usage().response_tokens)],
        }

    async def get_subtype_prevalence(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.subtype_prevalence.prompt.format(
                disease_name=state["disease_name"],
                synonyms_prompt=state["synonyms_prompt"],
            ),
            model_config=self.subtype_prevalence.model_config,
            pydantic_model=SubtypePrevalenceResult,
            system_prompt=self.subtype_prevalence.system_prompt,
        )

        return {
            "subtype_parent_disease": response.output.parent_disease,
            "primary_disease_world_prevalence_explanation": response.output.parent_disease_world_prevalence_explanation,
            "primary_disease_world_prevalence": response.output.parent_disease_world_prevalence,
            "subtype_percentage_occurrence_explanation": response.output.subtype_percentage_occurrence_explanation,
            "subtype_percentage_occurrence": response.output.subtype_percentage_occurrence,
            "unsure": response.output.unsure,
            "request_token_counter": [("subtype_parent_disease", response.usage().request_tokens)],
            "response_token_counter": [("subtype_parent_disease", response.usage().response_tokens)],
        }

    async def check_population_subgroup(self, state: State) -> State:
        if "subtype_percentage_occurrence_explanation" in state:
            explanation = (
                state["primary_disease_world_prevalence_explanation"]
                + state["subtype_percentage_occurrence_explanation"]
            )
        else:
            explanation = state["primary_disease_world_prevalence_explanation"]

        response = await get_llm_response(
            prompt=self.population_subgroup.prompt.format(primary_disease_explanation=explanation),
            model_config=self.population_subgroup.model_config,
            pydantic_model=PopulationSubgroupResult,
            system_prompt=self.population_subgroup.system_prompt,
        )

        return {
            "subgroup_specified_explanation": response.output.subgroup_specified_explanation,
            "subgroup_specified": response.output.subgroup_specified,
            "population_subgroup": response.output.population_subgroup,
            "population_subgroup_size": response.output.population_subgroup_size,
            "request_token_counter": [("population_subgroup", response.usage().request_tokens)],
            "response_token_counter": [("population_subgroup", response.usage().response_tokens)],
        }

    async def extract_prevalence_data(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.extract_prevalence.prompt.format(
                disease_world_prevalence=state["primary_disease_world_prevalence"]
            ),
            model_config=self.extract_prevalence.model_config,
            pydantic_model=ExtractPrevalenceData,
            system_prompt=self.extract_prevalence.system_prompt,
        )

        return {
            "prevalence_data_explanation": response.output.categorisation_explanation,
            "prevalence_data_type": response.output.category,
            "prevalence_data_numerator": response.output.numerator,
            "prevalence_data_denominator": response.output.denominator,
            "request_token_counter": [("extract_prevalence_data", response.usage().request_tokens)],
            "response_token_counter": [("extract_prevalence_data", response.usage().response_tokens)],
        }

    def calculate_prevalence_in_100k(self, state: State) -> State:
        # factor to add if disease is a subtype (i.e. prevalence is a percentage of a different primary disease)
        if state["disease_label"] == "disease-subtype":
            subtype_factor = state["subtype_percentage_occurrence"] / 100
        else:
            subtype_factor = 1

        # factor to add if disease is a subgroup (i.e. prevalence is a percentage of a different population)
        if state["subgroup_specified"]:
            if state["population_subgroup_size"]:
                subgroup_proportion = state["population_subgroup_size"] / 8e9
            else:
                subgroup_proportion = 1
        else:
            subgroup_proportion = 1

        if state["prevalence_data_type"] == "number":
            ## NB if a condition affects X number of people no additional adjustment is needed to account for the subgroup
            ## for example if a condition affects 300,000 live births globally, the prevalence would be
            ## as 300,000 / 8e9 * 100,000 = 3.75 in 100,000
            prevalence_in_100k = state["prevalence_data_numerator"] / 8e9 * 100_000 * subtype_factor
        else:
            prevalence_in_100k = (
                (state["prevalence_data_numerator"] / state["prevalence_data_denominator"])
                * subgroup_proportion
                * 100_000
                * subtype_factor
            )

        return {"prevalence_in_100k": prevalence_in_100k}

    def categorise_prevalence_in_100k(self, state: State) -> State:
        if state["prevalence_in_100k"] < 1:
            return {"disease_world_prevalence_subgraph": "<1 in 100,000"}
        elif state["prevalence_in_100k"] < 10:
            return {"disease_world_prevalence_subgraph": "1-9 in 100,000"}
        elif state["prevalence_in_100k"] < 100:
            return {"disease_world_prevalence_subgraph": "10-99 in 100,000"}
        elif state["prevalence_in_100k"] < 1000:
            return {"disease_world_prevalence_subgraph": "100-999 in 100,000"}
        elif state["prevalence_in_100k"] < 10000:
            return {"disease_world_prevalence_subgraph": "1,000-9,999 in 100,000"}
        else:
            return {"disease_world_prevalence_subgraph": f"{state['prevalence_in_100k']:.2f} in 100,000"}

    def compile_graph(self):
        graph = StateGraph(self.State)
        graph.add_node("initialise_state", self.initialise_state)
        graph.add_node("get_primary_prevalence", self.get_primary_prevalence)
        graph.add_node("get_subtype_prevalence", self.get_subtype_prevalence)
        graph.add_node("check_population_subgroup", self.check_population_subgroup)
        graph.add_node("extract_prevalence_data", self.extract_prevalence_data)
        graph.add_node("calculate_prevalence_in_100k", self.calculate_prevalence_in_100k)
        graph.add_node("categorise_prevalence_in_100k", self.categorise_prevalence_in_100k)

        graph.add_edge(START, "initialise_state")
        graph.add_conditional_edges(
            "initialise_state",
            self.check_disease_label,
            {
                "disease_is_primary": "get_primary_prevalence",
                "disease_is_subtype": "get_subtype_prevalence",
            },
        )
        graph.add_edge("get_primary_prevalence", "check_population_subgroup")
        graph.add_edge("get_subtype_prevalence", "check_population_subgroup")
        graph.add_edge("check_population_subgroup", "extract_prevalence_data")
        graph.add_edge("extract_prevalence_data", "calculate_prevalence_in_100k")
        graph.add_edge("calculate_prevalence_in_100k", "categorise_prevalence_in_100k")
        graph.add_edge("categorise_prevalence_in_100k", END)

        return graph.compile()

    def get_mermaid_graph(self, all_subgraphs: bool = False) -> Image:
        """Get the Mermaid graph for the disease prevalence graph"""
        try:
            mermaid_code = self.compile_graph().get_graph(xray=all_subgraphs).draw_mermaid()

            return mermaid_code
        except Exception as e:
            logger.error(f"Error generating Mermaid graph: {str(e)}")
            return None

    async def safe_invoke(self, disease_name: str, synonyms: str, disease_label: str) -> dict[str, Any]:
        return await self.invoke(
            {
                "disease_name": disease_name,
                "synonyms": synonyms,
                "disease_label": disease_label,
            }
        )

    async def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        try:
            final_state = await self.compile_graph().ainvoke(inputs)
            return final_state
        except Exception as e:
            logger.error(f"Error in invoke: {str(e)}")
            raise


class DiseasePrevalenceConsensusGraph(InvokableGraph):
    class State(TypedDict):
        # input
        disease_name: str
        synonyms: str
        synonyms_prompt: str
        disease_label: str

        # assets
        primary_disease_world_prevalence_explanation: Annotated[list[str], add]
        primary_disease_world_prevalence: Annotated[list[str], add]
        subtype_parent_disease: Annotated[list[str], add]
        subtype_percentage_occurrence_explanation: Annotated[list[str], add]
        subtype_percentage_occurrence: Annotated[list[float], add]
        unsure: Annotated[list[bool], add]
        subgroup_specified_explanation: Annotated[list[str], add]
        subgroup_specified: Annotated[list[bool], add]
        population_subgroup: Annotated[list[str], add]
        population_subgroup_size: Annotated[list[float], add]
        prevalence_data_explanation: Annotated[list[str], add]
        prevalence_data_type: Annotated[list[str], add]
        prevalence_data_numerator: Annotated[list[str], add]
        prevalence_data_denominator: Annotated[list[str], add]
        prevalence_in_100k: Annotated[list[float], add]
        disease_world_prevalence_subgraph: Annotated[list[str], add]
        experimental: bool

        # final response
        disease_world_prevalence: str

        # token counter
        request_token_counter: Annotated[list[tuple[int, str]], add]
        response_token_counter: Annotated[list[tuple[int, str]], add]

    def __init__(self, config: dict, aggregate_strategy: str = "mean", N: int = 3):
        self.config = config
        self.aggregate_strategy = aggregate_strategy
        self.N = N

        super().__init__()

    def check_disease_label(self, state: State) -> str:
        if state["disease_label"] == "disease-primary":
            return "disease_is_primary_or_subtype"
        elif state["disease_label"] == "disease-subtype":
            return "disease_is_primary_or_subtype"
        elif state["disease_label"] == "disease-iatrogenic/drug-induced":
            return "disease_is_iatrogenic_or_no_label"
        else:
            return "disease_is_no_label"

    def initialise_state(self, state: State) -> State:
        return {
            "request_token_counter": [],
            "response_token_counter": [],
            "experimental": True if not state["disease_label"] else False,
        }

    async def call_subgraph(self, state: State) -> State:
        subgraph = DiseasePrevalenceGraph(self.config)
        subgraph_response = await subgraph.safe_invoke(
            disease_name=state["disease_name"],
            synonyms=state["synonyms"],
            disease_label=state["disease_label"],
        )

        fields_to_drop = [
            "disease_name",
            "synonyms",
            "synonyms_prompt",
            "disease_label",
        ]
        filtered_response = {
            k: [v] if v is not None else [None] for k, v in subgraph_response.items() if k not in fields_to_drop
        }
        return filtered_response

    def aggregate_state(self, state: State) -> State:
        # options for other aggregate strategies left, in case we decide to change it in future
        # but testing indicates that using the mean with N=3 is a good compromise between accuracy and speed
        if self.aggregate_strategy == "mode":
            return {
                "disease_world_prevalence": Counter(state["disease_world_prevalence_subgraphs"]).most_common(1)[0][0]
            }
        elif self.aggregate_strategy == "lowest_denominator":
            # chose the item in disease_world_prevalence as the index of the lowest prevalence_in_100k
            return {
                "disease_world_prevalence": state["disease_world_prevalence_subgraphs"][
                    state["prevalence_in_100k"].index(min(state["prevalence_in_100k"]))
                ]
            }
        elif self.aggregate_strategy == "mean":
            mean = sum(state["prevalence_in_100k"]) / len(state["prevalence_in_100k"])
            if mean < 1:
                result = {"disease_world_prevalence": "<1 in 100,000"}
            elif mean < 10:
                result = {"disease_world_prevalence": "1-9 in 100,000"}
            elif mean < 100:
                result = {"disease_world_prevalence": "10-99 in 100,000"}
            elif mean < 1000:
                result = {"disease_world_prevalence": "100-999 in 100,000"}
            elif mean < 10000:
                result = {"disease_world_prevalence": "1,000-9,999 in 100,000"}
            # handle cases where the mean is greater than 100,000 -- usually caused by LLM errors
            elif mean > 100000:
                result = {"disease_world_prevalence": "UNCLEAR"}
            else:
                result = {"disease_world_prevalence": f"{mean:.2f} in 100,000"}
            return result
        else:
            raise ValueError(f"Invalid aggregate strategy: {self.aggregate_strategy}")

    def return_iatrogenic_result(self, state: State) -> State:
        return {
            "disease_world_prevalence": None,
            "primary_disease_world_prevalence_explanation": [
                "prevalence not generated because disease is iatrogenic or drug-induced"
            ],
        }

    def compile_graph(self):
        graph = StateGraph(self.State)
        graph.add_node("initialise_state", self.initialise_state)
        for i in range(self.N):  # add N subgraphs
            graph.add_node(f"subgraph_{i}", self.call_subgraph)
        graph.add_node("aggregate_state", self.aggregate_state, defer_node=True)
        graph.add_node("return_iatrogenic_result", self.return_iatrogenic_result)

        graph.add_conditional_edges(
            START,
            self.check_disease_label,
            {
                "disease_is_primary_or_subtype": "initialise_state",
                "disease_is_iatrogenic_or_no_label": "return_iatrogenic_result",
                "disease_is_no_label": "initialise_state",
            },
        )
        for i in range(self.N):
            graph.add_edge("initialise_state", f"subgraph_{i}")
            graph.add_edge(f"subgraph_{i}", "aggregate_state")
        graph.add_edge("aggregate_state", END)

        return graph.compile()

    async def safe_invoke(self, disease_name: str, synonyms: str, disease_label: str) -> dict[str, Any]:
        return await self.invoke(
            {
                "disease_name": disease_name,
                "synonyms": synonyms,
                "disease_label": disease_label,
            }
        )
