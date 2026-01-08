import logging
import re
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

from core_entities.utils.llm_utils import LLMConfig, get_llm_response

logger = logging.getLogger(__name__)
WHOCC_URL = "https://www.whocc.no/atc_ddd_index/?name={search_term}"

# TODO: move this to parameters
atc_main_system_prompt = "You work for the WHO Collaborating Centre for Drug Statistics Methodology. You are given a set of ATC codes for a drug, pick the one that you would consider to be the primary use of the drug. Use a single sentence in your explanations, don't use first person pronouns."
atc_main_prompt = """
The drug: "{drug_name}" has the following ATC associated: {atc_codes}. Pick the one that you would consider to be the primary use of the drug
"""
model_config = {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "max_output_tokens": 500,
}
choose_atm_main_llm_config = LLMConfig(
    {
        "system_prompt": atc_main_system_prompt,
        "prompt": atc_main_prompt,
        "model_config": model_config,
    }
)


def get_atc_from_whocc(search_term):
    """Get ATC code from WHO Collaborating Centre for Drug Statistics Methodology"""
    try:
        encoded_search_term = quote(search_term)
        url = WHOCC_URL.format(search_term=encoded_search_term)
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

            # Search results are returned in a table, if no table can be found, there are no search results, return empty list
            table = soup.find("table")
            if not table:
                raise Exception(
                    f"The search page for {search_term} does not contain a table, unable to find ATC code. This happened because there was no search results for {search_term}"
                )

            # Find all rows in the table
            rows = table.find_all("tr")
            atc_codes = []
            search_term_lower = search_term.lower().strip()

            # check that the table has 2 columns
            for row in rows:
                if len(row.find_all("td")) != 2:
                    raise Exception(
                        f"The search page for {search_term} returned a table with {len(row.find_all('td'))} columns, expected 2. This happened because there was no search results for {search_term}"
                    )

            # Go through the table row by row, and only pull codes if it's an exact match to the search term
            # and if the ATC code is in the correct format
            for row in rows:
                # Get all cells in the row
                cells = row.find_all("td")
                # first column: ATC code
                atc_code = cells[0].get_text(strip=True)

                # second column: drug name (inside a link)
                drug_name_cell = cells[1]
                link = drug_name_cell.find("a")
                drug_name_found = link.get_text(strip=True).lower()

                # Check for exact match (case-insensitive)
                if drug_name_found == search_term_lower:
                    # Validate ATC code format before adding
                    # All ATC codes start with a letter followed by 2 letters and 2 digits
                    if re.match(r"^[A-Z]\d{2}[A-Z]{2}\d{2}$", atc_code):
                        atc_codes.append(atc_code)

            return list(set(atc_codes)) if atc_codes else []
        else:
            raise Exception(
                f"The search page for {search_term} returned a status code {response.status_code}, unable to find ATC code"
            )

    except Exception as e:
        logger.error(f"Error getting ATC from WHO for {search_term}: {str(e)}")
        return None


async def get_atc_main_llm(drug_name, atc_codes):
    class RemappedReasonResult(BaseModel):
        explanation: str
        atc_main: str

    result = await get_llm_response(
        prompt=choose_atm_main_llm_config.prompt.format(drug_name=drug_name, atc_codes=atc_codes),
        model_config=choose_atm_main_llm_config.model_config,
        pydantic_model=RemappedReasonResult,
        system_prompt=choose_atm_main_llm_config.system_prompt,
    )

    return result.output.atc_main


async def get_drug_atc_codes(drug_name, synonyms):
    atc_name = get_atc_from_whocc(drug_name)

    # TODO: Loop over atc_synonyms to find ATC codes for all items
    atc_synonym = get_atc_from_whocc(synonyms) if synonyms else []

    all_atc = set(atc_name + atc_synonym)

    if len(all_atc) > 1:
        # choose the most suitable with an LLM
        atc_main = await get_atc_main_llm(drug_name, all_atc)
    elif len(all_atc) == 1:
        atc_main = all_atc.pop()
    else:
        atc_main = None

    return {
        "atc_name": None if not atc_name else atc_name,
        "atc_synonym": None if not atc_synonym else atc_synonym,
        "atc_main": None if not atc_main else atc_main,
    }


if __name__ == "__main__":
    import asyncio

    async def main():
        drug_name = "epinephrine"
        synonyms = "adrenaline"

        result = await get_drug_atc_codes(drug_name, synonyms)
        return result

    logger.info(asyncio.run(main()))
