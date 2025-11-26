import asyncio
import logging
from typing import Dict, List

import pandas as pd
import tqdm.asyncio
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from matrix_inject.inject import inject_object

logger = logging.getLogger(__name__)


class SetOutputParser:
    def __init__(self, allowed_outputs: List[str]):
        """
        Initialize the parser with a predefined set of allowed outputs.
        """
        self.allowed_outputs = set(allowed_outputs)

    def parse(self, text: str):
        """
        Parse the model output to ensure it's in the allowed set.
        """
        cleaned_text = text.strip()
        if cleaned_text in self.allowed_outputs:
            return cleaned_text
        else:
            raise ValueError(f"Output '{text}' is not in the allowed set: {self.allowed_outputs}")


class Tag:
    """Utility function to help with tag generation."""

    def __init__(self, output_col: str, prompt: str, output_parser: BaseTransformOutputParser) -> None:
        self._output_col = output_col
        self._prompt = prompt
        self._output_parser = output_parser

    def generate(
        self,
        loop: asyncio.BaseEventLoop,
        df: pd.DataFrame,
        model: BaseChatModel,
        max_workers: int = 100,
        timeout: int = 20,
    ):
        """
        Function to generate tags.

        Function generates tags for each row in the input dataframe in parallel.

        Args:
            loop: event loop to run on
            df: input dataframe
            model: model to apply
            max_workers: max workers to use
            timeout: max timeout for call
        """
        sem = asyncio.Semaphore(max_workers)
        tasks = [loop.create_task(self._process_row(sem, model, row.to_dict())) for _, row in sorted(df.iterrows())]

        # Track progress with tqdm as tasks complete
        results = []
        with tqdm.asyncio.tqdm(total=len(tasks), desc="Enriching elements") as progress_bar:

            async def monitor_tasks():
                for task in asyncio.as_completed(tasks):
                    try:
                        result = await asyncio.wait_for(task, timeout)
                        results.append(result)
                    except asyncio.TimeoutError as e:
                        logger.error(f"Timeout error: partition processing took longer than {timeout} seconds.")
                        raise e
                    except Exception as e:
                        logger.error(f"Error processing partition in tqdm loop: {e}")
                        raise e
                    finally:
                        progress_bar.update(1)

            # Run the monitoring coroutine
            loop.run_until_complete(monitor_tasks())

        df[self._output_col] = results
        return df

    async def _process_row(self, sem: asyncio.Semaphore, model: BaseChatModel, row: Dict):
        async with sem:
            prompt = ChatPromptTemplate.from_messages([HumanMessage(content=self._prompt.format(**row))])
            response = await model.ainvoke(prompt.format_messages())
            return self._output_parser.parse(response.content)


@inject_object()
def generate_tags(df: pd.DataFrame, model: BaseChatModel, tags: List[Tag]) -> List:
    """Function to generate tags based on provided prompts and params through LLM

    Args:
        df: DataFrame - input dataframe
        model: BaseChatModel - model to apply
        tags: List[Tag] - List of tags to generate
    Returns
        Enriched df
    """

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        for tag in tags:
            df = tag.generate(loop, df, model)
    finally:
        loop.close()

    return df
