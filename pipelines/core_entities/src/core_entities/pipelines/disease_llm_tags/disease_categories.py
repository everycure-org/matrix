from collections import Counter
from functools import partial
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from core_entities.utils.llm_utils import InvokableGraph, LLMConfig, get_llm_response

#########################################################
# Pydantic Models
#########################################################


class DefaultDiseaseCategoryClassification(BaseModel):
    explanation: str
    classification: bool


#########################################################
# Disease Category Graph - With N repeats to find consensus
#########################################################


class DiseaseCategoryGraph(InvokableGraph):
    class State(TypedDict):
        # input
        entity: str
        synonyms: str | None

        # synonym prompt
        synonym_prompt: str

        # final results
        results: Annotated[list[tuple[str, bool]], add]  # explanation, classification
        all_explanations: list[str]
        all_classifications: list[bool]
        final_explanation: str
        final_classification: bool
        request_token_counter: Annotated[list[tuple[int, str]], add]
        response_token_counter: Annotated[list[tuple[int, str]], add]

    @classmethod
    def from_config(
        cls,
        config: dict,
        disease_category: str,
        pydantic_model=DefaultDiseaseCategoryClassification,
    ):
        return cls(config, disease_category, pydantic_model)

    def __init__(
        self,
        config: dict,
        disease_category: str,
        pydantic_model=DefaultDiseaseCategoryClassification,
    ):
        self.config = config
        # number of iterations to find consensus
        self.N = config.get("N")
        self._setup_config_attributes(disease_category, pydantic_model)
        super().__init__()

    def _setup_config_attributes(self, disease_category: str, pydantic_model: BaseModel):
        self.llm_config = LLMConfig(self.config.get(disease_category))
        self.pydantic_model = pydantic_model
        self.disease_category = disease_category

    def initialise_state(self, state: State) -> dict:
        return {
            "results": [],
            "request_token_counter": [],
            "response_token_counter": [],
            "synonym_prompt": f"This entity is also known as {state['synonyms']}" if state["synonyms"] else "",
        }

    async def get_disease_category_classification(
        self,
        state: State,
        disease_category: str,
        pydantic_model: DefaultDiseaseCategoryClassification,
        iteration: int,
    ) -> dict:
        response = await get_llm_response(
            prompt=self.llm_config.prompt.format(disease=state["entity"], synonym_prompt=state["synonym_prompt"]),
            model_config=self.llm_config.model_config,
            pydantic_model=pydantic_model,
            system_prompt=self.llm_config.system_prompt,
        )

        return {
            "results": [(response.output.explanation, response.output.classification)],
            "request_token_counter": [(f"{disease_category}_{iteration}", response.usage().request_tokens)],
            "response_token_counter": [(f"{disease_category}_{iteration}", response.usage().response_tokens)],
        }

    def retrieve_consensus(self, state: State) -> dict:
        def _get_mode(results: list[tuple[str, bool]]) -> bool:
            # Extract just the boolean values from each dict
            boolean_values = [result[1] for result in results]

            # Use Counter to find the mode
            counter = Counter(boolean_values)
            mode = counter.most_common(1)[0][0]  # Get the most common boolean
            return mode

        def _get_explanation(results: list[tuple[str, bool]], mode: bool) -> str:
            for result in results:
                if result[1] == mode:
                    explanation = result[0]
                    break

            return explanation

        return {
            "final_explanation": _get_explanation(state["results"], _get_mode(state["results"])),
            "final_classification": _get_mode(state["results"]),
            "all_explanations": [result[0] for result in state["results"]],
            "all_classifications": [result[1] for result in state["results"]],
        }

    def compile_graph(self) -> StateGraph:
        graph = StateGraph(self.State)
        graph.add_node("initialise_state", self.initialise_state)
        for i in range(self.N):
            graph.add_node(
                f"disease_subgraph_{i}",
                partial(
                    self.get_disease_category_classification,
                    disease_category=self.disease_category,
                    pydantic_model=self.pydantic_model,
                    iteration=i,
                ),
            )
        graph.add_node("retrieve_consensus", self.retrieve_consensus)

        graph.add_edge(START, "initialise_state")
        for i in range(self.N):
            graph.add_edge("initialise_state", f"disease_subgraph_{i}")
            graph.add_edge(f"disease_subgraph_{i}", "retrieve_consensus")
        graph.add_edge("retrieve_consensus", END)

        return graph.compile()

    async def safe_invoke(self, disease_name: str, synonyms: str) -> dict[str, Any]:
        return await self.compile_graph().ainvoke({"entity": disease_name, "synonyms": synonyms})


#########################################################
# Disease Categories Graph - Repeat Disease Category Graph over all categories
#########################################################


class DiseaseCategoriesGraphWithConsensus(InvokableGraph):
    class State(TypedDict):
        # input
        entity: str
        synonyms: str | None

        # aggregate results from all iterations of the disease category graph for each disease category
        results: dict[str, dict[str, list[str]]]

        # final results after consensus
        is_psychiatric_disease: bool
        psychiatric_disease_explanation: str
        is_malignant_cancer: bool
        malignant_cancer_explanation: str
        is_benign_tumour: bool
        benign_tumour_explanation: str
        is_pathogen_caused: bool
        pathogen_caused_explanation: str
        is_glucose_dysfunction: bool
        glucose_dysfunction_explanation: str

        # token counter
        request_token_counter: Annotated[list[tuple[int, str]], add]
        response_token_counter: Annotated[list[tuple[int, str]], add]

    def __init__(self, config: dict):
        self.config = config
        self.disease_categories = config.get("disease_categories")
        super().__init__()

    def initialise_state(self, state: State) -> dict:
        return {
            "request_token_counter": [],
            "response_token_counter": [],
            "results": {category: {"explanation": [], "classification": []} for category in self.disease_categories},
        }

    async def call_subgraph(self, state: State, disease_category: str) -> dict:
        subgraph = DiseaseCategoryGraph.from_config(self.config, disease_category)
        subgraph_response = await subgraph.safe_invoke(state["entity"], state["synonyms"])

        state["results"][disease_category]["explanation"] = subgraph_response["all_explanations"]
        state["results"][disease_category]["classification"] = subgraph_response["all_classifications"]

        return {
            f"is_{disease_category}": subgraph_response["final_classification"],
            f"{disease_category}_explanation": subgraph_response["final_explanation"],
            "request_token_counter": subgraph_response["request_token_counter"],
            "response_token_counter": subgraph_response["response_token_counter"],
        }

    def defer_nodes(self, state: State) -> State:
        return {}

    def compile_graph(self):
        graph = StateGraph(self.State)
        graph.add_node("initialise_state", self.initialise_state)
        for disease_category in self.disease_categories:
            graph.add_node(
                f"{disease_category}_subgraph",
                partial(self.call_subgraph, disease_category=disease_category),
            )
        graph.add_node("defer_nodes", self.defer_nodes)

        graph.add_edge(START, "initialise_state")
        for disease_category in self.disease_categories:
            graph.add_edge("initialise_state", f"{disease_category}_subgraph")
            graph.add_edge(f"{disease_category}_subgraph", "defer_nodes")
        graph.add_edge("defer_nodes", END)

        return graph.compile()

    async def safe_invoke(self, disease_name: str, synonyms: str) -> dict[str, Any]:
        return await self.invoke({"entity": disease_name, "synonyms": synonyms})
