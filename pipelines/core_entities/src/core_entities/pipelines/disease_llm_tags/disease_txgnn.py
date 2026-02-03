from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from core_entities.utils.llm_utils import InvokableGraph, LLMConfig, get_llm_response


class TxGNNClassification(BaseModel):
    """Pydantic model for TXGNN classification output."""

    explanation: str = Field(description="Brief explanation of the classification reasoning")
    categories: list[str] = Field(description="List of best-fitting disease categories from the allowed set")


class DiseaseTxGNNGraph(InvokableGraph):
    """Graph for classifying diseases into TXGNN categories.

    TXGNN categories are designed to classify diseases into clinically relevant
    therapeutic categories for drug repurposing analysis.
    """

    class State(TypedDict):
        # input
        entity: str
        synonyms: str | None

        # prompt construction
        synonym_prompt: str
        categories_list: str

        # results
        txgnn: str
        txgnn_explanation: str
        request_token_counter: Annotated[list[tuple[str, int]], list]
        response_token_counter: Annotated[list[tuple[str, int]], list]

    def __init__(self, config: dict):
        self.config = config
        self.llm_config = LLMConfig(config)
        self.categories = config.get("categories", [])
        super().__init__()

    def initialise_state(self, state: State) -> dict:
        return {
            "request_token_counter": [],
            "response_token_counter": [],
            "synonym_prompt": f"This disease is also known as: {state['synonyms']}" if state["synonyms"] else "",
            "categories_list": ", ".join(self.categories),
        }

    async def classify_disease(self, state: State) -> dict:
        prompt = self.llm_config.prompt.format(
            disease=state["entity"],
            synonym_prompt=state["synonym_prompt"],
            categories_list=state["categories_list"],
        )

        response = await get_llm_response(
            prompt=prompt,
            model_config=self.llm_config.model_config,
            pydantic_model=TxGNNClassification,
            system_prompt=self.llm_config.system_prompt,
        )

        # Filter categories to only include valid ones
        valid_categories = [cat for cat in response.output.categories if cat in self.categories]

        # If no valid categories found, default to "other"
        if not valid_categories:
            valid_categories = ["other"]

        # Join categories with pipe delimiter
        txgnn_value = "|".join(valid_categories)

        return {
            "txgnn": txgnn_value,
            "txgnn_explanation": response.output.explanation,
            "request_token_counter": [("txgnn", response.usage().request_tokens)],
            "response_token_counter": [("txgnn", response.usage().response_tokens)],
        }

    def compile_graph(self) -> StateGraph:
        graph = StateGraph(self.State)
        graph.add_node("initialise_state", self.initialise_state)
        graph.add_node("classify_disease", self.classify_disease)

        graph.add_edge(START, "initialise_state")
        graph.add_edge("initialise_state", "classify_disease")
        graph.add_edge("classify_disease", END)

        return graph.compile()

    async def safe_invoke(self, disease_name: str, synonyms: str) -> dict[str, Any]:
        return await self.invoke({"entity": disease_name, "synonyms": synonyms})
