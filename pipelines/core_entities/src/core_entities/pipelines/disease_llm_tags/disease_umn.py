from operator import add
from typing import Annotated, Any, Literal, TypedDict

from IPython.display import Image
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from core_entities.utils.llm_utils import InvokableGraph, LLMConfig, get_llm_response


# intialise pydantic models for unmet medical need
class DefaultFeatureResponse(BaseModel):
    reasoning: str
    category: Literal[1, 2, 3, 4, 5]


class RegulationBarriersResponse(BaseModel):
    reasoning: str
    category: Literal[1, 3, 5]


# define the unmet medical need graph
class DiseaseUMNGraph(InvokableGraph):
    class State(TypedDict):
        # input
        entity: str
        synonyms: str

        # assets
        synonym_prompt: str

        # feature assets
        commonality: int
        commonality_explanation: str
        duration: int
        duration_explanation: str
        QALY: int
        QALY_explanation: str
        mortality: int
        mortality_explanation: str
        disease_modification: int
        disease_modification_explanation: str
        adverse_effects: int
        adverse_effects_explanation: str
        route_of_administration: int
        route_of_administration_explanation: str
        frequency_of_administration: int
        frequency_of_administration_explanation: str
        cost_to_patients: int
        cost_to_patients_explanation: str
        robust_supply: int
        robust_supply_explanation: str
        regulation_barriers: int
        regulation_barriers_explanation: str
        umn_score: float

        # token counter
        request_token_counter: Annotated[list[tuple[int, str]], add]
        response_token_counter: Annotated[list[tuple[int, str]], add]

    def __init__(self, config: dict):
        self.config = config
        self._setup_config_attributes()

        super().__init__()

    def _setup_config_attributes(self):
        self.commonality = LLMConfig(self.config.get("commonality"))
        self.duration = LLMConfig(self.config.get("duration"))
        self.QALY = LLMConfig(self.config.get("QALY"))
        self.mortality = LLMConfig(self.config.get("mortality"))
        self.disease_modification = LLMConfig(self.config.get("disease_modification"))
        self.adverse_effects = LLMConfig(self.config.get("adverse_effects"))
        self.route_of_administration = LLMConfig(self.config.get("route_of_administration"))
        self.frequency_of_administration = LLMConfig(self.config.get("frequency_of_administration"))
        self.cost_to_patients = LLMConfig(self.config.get("cost_to_patients"))
        self.robust_supply = LLMConfig(self.config.get("robust_supply"))
        self.regulation_barriers = LLMConfig(self.config.get("regulation_barriers"))
        # NB:generic patent queries stored as a record, but is not currently calculated or used in this pipeline
        self.generic_patent_1 = LLMConfig(self.config.get("generic_patent_1"))
        self.generic_patent_2 = LLMConfig(self.config.get("generic_patent_2"))

    def initialise_state(self, state: State) -> State:
        return {
            "request_token_counter": [],
            "response_token_counter": [],
            "synonym_prompt": f"(This entity is also known as {state['synonyms']})" if state["synonyms"] else "",
        }

    async def get_commonality(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.commonality.prompt.format(entity=state["entity"], synonym_prompt=state["synonym_prompt"]),
            model_config=self.commonality.model_config,
            pydantic_model=DefaultFeatureResponse,
            system_prompt=self.commonality.system_prompt,
        )

        return {
            "commonality": response.output.category,
            "commonality_explanation": response.output.reasoning,
            "request_token_counter": [("commonality", response.usage().request_tokens)],
            "response_token_counter": [("commonality", response.usage().response_tokens)],
        }

    async def get_duration(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.duration.prompt.format(entity=state["entity"], synonym_prompt=state["synonym_prompt"]),
            model_config=self.duration.model_config,
            pydantic_model=DefaultFeatureResponse,
            system_prompt=self.duration.system_prompt,
        )

        return {
            "duration": response.output.category,
            "duration_explanation": response.output.reasoning,
            "request_token_counter": [("duration", response.usage().request_tokens)],
            "response_token_counter": [("duration", response.usage().response_tokens)],
        }

    async def get_QALY(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.QALY.prompt.format(entity=state["entity"], synonym_prompt=state["synonym_prompt"]),
            model_config=self.QALY.model_config,
            pydantic_model=DefaultFeatureResponse,
            system_prompt=self.QALY.system_prompt,
        )

        return {
            "QALY": response.output.category,
            "QALY_explanation": response.output.reasoning,
            "request_token_counter": [("QALY", response.usage().request_tokens)],
            "response_token_counter": [("QALY", response.usage().response_tokens)],
        }

    async def get_mortality(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.mortality.prompt.format(entity=state["entity"], synonym_prompt=state["synonym_prompt"]),
            model_config=self.mortality.model_config,
            pydantic_model=DefaultFeatureResponse,
            system_prompt=self.mortality.system_prompt,
        )

        return {
            "mortality": response.output.category,
            "mortality_explanation": response.output.reasoning,
            "request_token_counter": [("mortality", response.usage().request_tokens)],
            "response_token_counter": [("mortality", response.usage().response_tokens)],
        }

    async def get_disease_modification(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.disease_modification.prompt.format(
                entity=state["entity"], synonym_prompt=state["synonym_prompt"]
            ),
            model_config=self.disease_modification.model_config,
            pydantic_model=DefaultFeatureResponse,
            system_prompt=self.disease_modification.system_prompt,
        )

        return {
            "disease_modification": response.output.category,
            "disease_modification_explanation": response.output.reasoning,
            "request_token_counter": [("disease_modification", response.usage().request_tokens)],
            "response_token_counter": [("disease_modification", response.usage().response_tokens)],
        }

    async def get_adverse_effects(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.adverse_effects.prompt.format(entity=state["entity"], synonym_prompt=state["synonym_prompt"]),
            model_config=self.adverse_effects.model_config,
            pydantic_model=DefaultFeatureResponse,
            system_prompt=self.adverse_effects.system_prompt,
        )

        return {
            "adverse_effects": response.output.category,
            "adverse_effects_explanation": response.output.reasoning,
            "request_token_counter": [("adverse_effects", response.usage().request_tokens)],
            "response_token_counter": [("adverse_effects", response.usage().response_tokens)],
        }

    async def get_route_of_administration(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.route_of_administration.prompt.format(
                entity=state["entity"], synonym_prompt=state["synonym_prompt"]
            ),
            model_config=self.route_of_administration.model_config,
            pydantic_model=DefaultFeatureResponse,
            system_prompt=self.route_of_administration.system_prompt,
        )

        return {
            "route_of_administration": response.output.category,
            "route_of_administration_explanation": response.output.reasoning,
            "request_token_counter": [("route_of_administration", response.usage().request_tokens)],
            "response_token_counter": [("route_of_administration", response.usage().response_tokens)],
        }

    async def get_frequency_of_administration(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.frequency_of_administration.prompt.format(
                entity=state["entity"], synonym_prompt=state["synonym_prompt"]
            ),
            model_config=self.frequency_of_administration.model_config,
            pydantic_model=DefaultFeatureResponse,
            system_prompt=self.frequency_of_administration.system_prompt,
        )

        return {
            "frequency_of_administration": response.output.category,
            "frequency_of_administration_explanation": response.output.reasoning,
            "request_token_counter": [("frequency_of_administration", response.usage().request_tokens)],
            "response_token_counter": [("frequency_of_administration", response.usage().response_tokens)],
        }

    async def get_cost_to_patients(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.cost_to_patients.prompt.format(entity=state["entity"], synonym_prompt=state["synonym_prompt"]),
            model_config=self.cost_to_patients.model_config,
            pydantic_model=DefaultFeatureResponse,
            system_prompt=self.cost_to_patients.system_prompt,
        )

        return {
            "cost_to_patients": response.output.category,
            "cost_to_patients_explanation": response.output.reasoning,
            "request_token_counter": [("cost_to_patients", response.usage().request_tokens)],
            "response_token_counter": [("cost_to_patients", response.usage().response_tokens)],
        }

    async def get_robust_supply(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.robust_supply.prompt.format(entity=state["entity"], synonym_prompt=state["synonym_prompt"]),
            model_config=self.robust_supply.model_config,
            pydantic_model=DefaultFeatureResponse,
            system_prompt=self.robust_supply.system_prompt,
        )

        return {
            "robust_supply": response.output.category,
            "robust_supply_explanation": response.output.reasoning,
            "request_token_counter": [("robust_supply", response.usage().request_tokens)],
            "response_token_counter": [("robust_supply", response.usage().response_tokens)],
        }

    async def get_regulation_barriers(self, state: State) -> State:
        response = await get_llm_response(
            prompt=self.regulation_barriers.prompt.format(
                entity=state["entity"], synonym_prompt=state["synonym_prompt"]
            ),
            model_config=self.regulation_barriers.model_config,
            pydantic_model=RegulationBarriersResponse,
            system_prompt=self.regulation_barriers.system_prompt,
        )

        return {
            "regulation_barriers": response.output.category,
            "regulation_barriers_explanation": response.output.reasoning,
            "request_token_counter": [("regulation_barriers", response.usage().request_tokens)],
            "response_token_counter": [("regulation_barriers", response.usage().response_tokens)],
        }

    def calculate_umn_score(self, state: State) -> State:
        weights = self.config.get("unmet_medical_need_weights")
        umn_score = sum(state[feature] * weights[feature] for feature in weights)
        return {
            "umn_score": umn_score,
        }

    def compile_graph(self):
        main_graph = StateGraph(self.State)

        # add nodes
        main_graph.add_node("initialise_state", self.initialise_state)
        main_graph.add_node("get_commonality", self.get_commonality)
        main_graph.add_node("get_duration", self.get_duration)
        main_graph.add_node("get_QALY", self.get_QALY)
        main_graph.add_node("get_mortality", self.get_mortality)
        main_graph.add_node("get_disease_modification", self.get_disease_modification)
        main_graph.add_node("get_adverse_effects", self.get_adverse_effects)
        main_graph.add_node("get_route_of_administration", self.get_route_of_administration)
        main_graph.add_node("get_frequency_of_administration", self.get_frequency_of_administration)
        main_graph.add_node("get_cost_to_patients", self.get_cost_to_patients)
        main_graph.add_node("get_robust_supply", self.get_robust_supply)
        main_graph.add_node("get_regulation_barriers", self.get_regulation_barriers)
        main_graph.add_node("calculate_umn_score", self.calculate_umn_score, defer_node=True)

        # add edges
        main_graph.add_edge(START, "initialise_state")
        main_graph.add_edge("initialise_state", "get_commonality")
        main_graph.add_edge("initialise_state", "get_duration")
        main_graph.add_edge("initialise_state", "get_QALY")
        main_graph.add_edge("initialise_state", "get_mortality")
        main_graph.add_edge("initialise_state", "get_disease_modification")
        main_graph.add_edge("initialise_state", "get_adverse_effects")
        main_graph.add_edge("initialise_state", "get_route_of_administration")
        main_graph.add_edge("initialise_state", "get_frequency_of_administration")
        main_graph.add_edge("initialise_state", "get_cost_to_patients")
        main_graph.add_edge("initialise_state", "get_robust_supply")
        main_graph.add_edge("initialise_state", "get_regulation_barriers")
        main_graph.add_edge("get_commonality", "calculate_umn_score")
        main_graph.add_edge("get_duration", "calculate_umn_score")
        main_graph.add_edge("get_QALY", "calculate_umn_score")
        main_graph.add_edge("get_mortality", "calculate_umn_score")
        main_graph.add_edge("get_disease_modification", "calculate_umn_score")
        main_graph.add_edge("get_adverse_effects", "calculate_umn_score")
        main_graph.add_edge("get_route_of_administration", "calculate_umn_score")
        main_graph.add_edge("get_frequency_of_administration", "calculate_umn_score")
        main_graph.add_edge("get_cost_to_patients", "calculate_umn_score")
        main_graph.add_edge("get_robust_supply", "calculate_umn_score")
        main_graph.add_edge("get_regulation_barriers", "calculate_umn_score")
        main_graph.add_edge("calculate_umn_score", END)

        return main_graph.compile()

    def get_mermaid_graph(self, all_subgraphs: bool = True):
        return Image(
            self._compiled_graph.get_graph(xray=all_subgraphs).draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
        )

    async def safe_invoke(self, disease_name: str, synonyms: str) -> dict[str, Any]:
        return await self.invoke({"entity": disease_name, "synonyms": synonyms})
