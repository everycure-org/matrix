from collections import Counter
from operator import add
from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from core_entities.utils.llm_utils import InvokableGraph, LLMConfig, get_llm_response


# initialise pydantic models for entity classification
class LabelJustification(BaseModel):
    labels: Literal[
        "drug_amenable",
        "gene_therapy_candidate",
        "structural",
        "behavioral_phychosocial_nutritional",
        "umbrella_non_specific",
        "clinical_findings",
        "non_life_threatening",
    ]
    justification: str
    selected: bool


class LabelEntityResponse(BaseModel):
    label_justifications: list[LabelJustification]
    entity_labels_explanation: str
    entity_final_labels: list[str]


# initialise pydantic models for disease labelling
class LabelDiseaseResponse(BaseModel):
    label: Literal["disease-primary", "disease-subtype", "disease-iatrogenic/drug-induced"]
    explanation: str


class DiseaseLabelsGraph(InvokableGraph):
    class State(TypedDict):
        entity: str
        synonyms: str
        synonyms_prompt: str
        entity_labels_all: Annotated[list[str], add]
        entity_labels_judgements: Annotated[list[dict], add]
        entity_labels: list[str]
        entity_labels_explanation: dict[str, str]
        disease_label_all: Annotated[list[str], add]
        disease_label_judgements: Annotated[list[dict], add]
        disease_label: list[str]
        disease_label_explanation: str
        request_token_counter: Annotated[list[tuple[int, str]], add]
        response_token_counter: Annotated[list[tuple[int, str]], add]

    def __init__(self, config: dict):
        self.config = config
        self._setup_config_attributes()
        super().__init__()

    def _setup_config_attributes(self):
        self.entity_label = LLMConfig(self.config.get("entity_label"))
        self.disease_label = LLMConfig(self.config.get("disease_label"))

    @staticmethod
    def initialise_state(state: State) -> dict:
        return {
            "synonyms_prompt": f"It is also known as {state['synonyms']}." if state["synonyms"] else "",
            "request_token_counter": [],
            "response_token_counter": [],
            "entity_labels": [],
            "disease_label": None,
        }

    async def get_entity_labels(self, state: State) -> dict:
        response = await get_llm_response(
            prompt=self.entity_label.prompt.format(entity=state["entity"], synonym_prompt=state["synonyms_prompt"]),
            model_config=self.entity_label.model_config,
            pydantic_model=LabelEntityResponse,
            system_prompt=self.entity_label.system_prompt,
        )

        return {
            "entity_labels_all": response.output.entity_final_labels,
            "entity_labels_judgements": response.output.label_justifications,
            "request_token_counter": [("entity_label", response.usage().request_tokens)],
            "response_token_counter": [("entity_label", response.usage().response_tokens)],
        }

    async def get_disease_label(self, state: State) -> dict:
        response = await get_llm_response(
            prompt=self.disease_label.prompt.format(entity=state["entity"]),
            model_config=self.disease_label.model_config,
            pydantic_model=LabelDiseaseResponse,
            system_prompt=self.disease_label.system_prompt,
        )
        return {
            "disease_label_all": [response.output.label],
            "disease_label_judgements": [response.output],
            "request_token_counter": [("disease_label", response.usage().request_tokens)],
            "response_token_counter": [("disease_label", response.usage().response_tokens)],
        }

    def get_entity_consensus(self, state: State) -> dict:
        # aggregate all entity labels and get unique labels only
        entity_labels = list(set(state["entity_labels_all"]))

        label_justifications = {}

        # for each unique entity label, find the justification that matches the selected label
        # this is a naive approach to obtain a relavant justification for each label by selecting the first one that matches
        for label in entity_labels:
            justification = next(
                (j for j in state["entity_labels_judgements"] if j.labels == label and j.selected),
                None,
            )
            if justification:
                label_justifications[label] = justification.justification
            else:
                raise ValueError(f"No justification found for label: {label}")

        return {
            "entity_labels": entity_labels,
            "entity_labels_explanation": label_justifications,
        }

    def get_disease_consensus(self, state: State) -> dict:
        # Counter.most_common(1) returns [(value, count)] - use [0][1] to get the number of times the most common value appears
        # since this procedure is repeated 3 times, the most common value will need to have a count >2 to be considered a consensus
        if Counter(state["disease_label_all"]).most_common(1)[0][1] > 2:
            mode = Counter(state["disease_label_all"]).most_common(1)[0][0]
        else:
            # there is no consensus, default to disease-primary
            mode = "disease-primary"

        # this is a naive approach to obtain a relavant justification where the disease label matches the mode
        justification = next((j for j in state["disease_label_judgements"] if j.label == mode), None)
        if justification:
            disease_label_justification = justification.explanation
        else:
            raise ValueError(f"No justification found for label: {mode}")

        return {
            "disease_label": mode,
            "disease_label_explanation": disease_label_justification,
        }

    def route_entity_labels(self, state: State) -> State:
        if "drug_amenable" in state["entity_labels"]:
            return "drug_amenable"
        else:
            return "not_drug_amenable"

    @staticmethod
    def defer_nodes() -> State:
        """Static method to buffer between entity labels and disease labels"""
        return {}

    def compile_graph(self):
        main_graph = StateGraph(self.State)

        # Add nodes
        main_graph.add_node("initialise_state", self.initialise_state)
        main_graph.add_node("get_entity_labels_1", self.get_entity_labels)
        main_graph.add_node("get_entity_labels_2", self.get_entity_labels)
        main_graph.add_node("get_entity_labels_3", self.get_entity_labels)
        main_graph.add_node("get_entity_consensus", self.get_entity_consensus, defer_node=True)
        main_graph.add_node("defer_nodes", self.defer_nodes)
        main_graph.add_node("get_disease_label_1", self.get_disease_label)
        main_graph.add_node("get_disease_label_2", self.get_disease_label)
        main_graph.add_node("get_disease_label_3", self.get_disease_label)
        main_graph.add_node("get_disease_consensus", self.get_disease_consensus, defer_node=True)

        # Add edges
        main_graph.add_edge(START, "initialise_state")
        main_graph.add_edge("initialise_state", "get_entity_labels_1")
        main_graph.add_edge("initialise_state", "get_entity_labels_2")
        main_graph.add_edge("initialise_state", "get_entity_labels_3")
        main_graph.add_edge("get_entity_labels_1", "get_entity_consensus")
        main_graph.add_edge("get_entity_labels_2", "get_entity_consensus")
        main_graph.add_edge("get_entity_labels_3", "get_entity_consensus")
        main_graph.add_conditional_edges(
            "get_entity_consensus",
            self.route_entity_labels,
            {"drug_amenable": "defer_nodes", "not_drug_amenable": END},
        )
        main_graph.add_edge("defer_nodes", "get_disease_label_1")
        main_graph.add_edge("defer_nodes", "get_disease_label_2")
        main_graph.add_edge("defer_nodes", "get_disease_label_3")
        main_graph.add_edge("get_disease_label_1", "get_disease_consensus")
        main_graph.add_edge("get_disease_label_2", "get_disease_consensus")
        main_graph.add_edge("get_disease_label_3", "get_disease_consensus")
        main_graph.add_edge("get_disease_consensus", END)

        return main_graph.compile()

    def post_process(self, state: State) -> dict:
        """Extract only the relevant keys from the final state"""
        keys_to_keep = [
            "entity",
            "entity_labels",
            "entity_labels_explanation",
            "disease_label",
            "disease_label_explanation",
            "request_token_counter",
            "response_token_counter",
        ]
        return {key: state[key] for key in keys_to_keep if key in state}

    async def safe_invoke(self, disease_name: str, synonyms: str) -> dict:
        final_state = await self.invoke({"entity": disease_name, "synonyms": synonyms})
        return self.post_process(final_state)
