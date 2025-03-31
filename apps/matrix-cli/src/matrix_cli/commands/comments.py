#!/bin/python
import json
import logging
import os
from enum import Enum
from typing import List, Optional, Set

import pandas as pd
import typer
from dotenv import load_dotenv
from github import Github
from github.Issue import Issue
from github.IssueComment import IssueComment
from github.Repository import Repository
from google import genai  # type: ignore
from google.genai import types  # type: ignore
from joblib import Memory, Parallel, delayed  # type: ignore
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.rich import tqdm
from urllib3.exceptions import NameResolutionError

from matrix_cli.components.settings import settings as cli_settings

# Need to ensure this happens before we initialize the genai client
load_dotenv()

app = typer.Typer(help="Github comment-related utility commands", no_args_is_help=True)
logging.basicConfig(level=logging.WARNING)
memory = Memory(location=".cache", verbose=0)
console = Console()

gemini = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)


class Classification(str, Enum):
    RISK = "risk"
    CONCERNING = "concerning"
    UNCLASSIFIED = "unclassified"
    NEUTRAL = "neutral"


class CommentEvaluation(BaseModel):
    content: str
    link: str
    author: str
    classification: Optional[Classification] = None
    extract: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.link))

    def __str__(self) -> str:
        return json.dumps(self.model_dump(), indent=2)

    def short_str(self) -> str:
        j = self.model_dump()
        del j["content"]
        return json.dumps(j, indent=2)


class CommentEvaluationSettings(BaseSettings):
    first_pass_prompt: str = """
    Your objective is the detection of harmful comments on our github as we prepare for
    an open sourcing of the repository.
    Please identify each comment as "neutral", "concerning" or "identified risk".

    - Neutral: Comments that are just focused on the topic or are otherwise not noteworthy
    - Concerning: Comments that may be misinterpreted, could be perceived as hostile or
      are unprofessional
    - Identified Risk: Comments that clearly break common code of conducts or are unfair,
      mean or otherwise risky to be put online.

    A few examples of comments:
    - Neutral: "wops yer rite"
    - Neutral: "Docstring args don't match"
    - Neutral: "Let's not discuss this here"
    - Neutral: "This is sick" (not a risk, the programmer is congratulating their colleague)
    - Neutral: "Am I missing something?"
    - Neutral: "I'm thinking this is too specific tbh."
    - Neutral: "Needs to be removed" (not a risk, but a direction from a maintainer)
    - Neutral: "This should not exist" (not a risk but a direct statement, albeit negative)
    - Concerning: "I'm not sure if you're aware, but your code is full of bugs"
    - risk: "You are a bad programmer"
    - risk: "What an idiot"


    Overall the bar should be that a concerning comment is one that can be perceived as hostile and
    a risk is a straight out mean or aggressive statement.

    Anker yourself around common code of conducts of major open source projects. Also note
    programmers are trying to get stuff done, not giving a perfect speech. Thus, some
    banter is fine.
    """
    second_pass_prompt: str = """
    You are given a list of comments that a previous screening has classified as concerning or risk.
    Please now go through this list and tell us which of these comments truly are a risk to our
    reputation as a non profit organisation open sourcing code. Use standard code of conducts of
    major open source projects to make your decision. Note that we do not care about any security
    risk issues as these are just comments but we care about conduct, tone and professionalism.

    Please return a list of comments that you think truly are a risk to our reputation. Return
    a list of json objects with link & classification.
    """
    model: str = cli_settings.base_model
    advanced_model: str = "gemini-2.5-pro-exp-03-25"


comment_settings = CommentEvaluationSettings()


@app.command()
def detect_risk(
    ids: str = typer.Option(
        help="The issues or PRs to evaluate. Comma separated list", default=None
    ),
    github_token: str = typer.Option(help="The Github token to use", envvar="GITHUB_TOKEN"),
    repo_id: str = typer.Option(help="The repo id to use", default="everycure-org/matrix"),
    prompt: str = typer.Option(
        help="The prompt to use", default=comment_settings.first_pass_prompt
    ),
    limit: Optional[int] = typer.Option(help="The limit of comments to evaluate", default=None),
    excel_output: str = typer.Option(
        help="The path to the excel file to save the results", default=None
    ),
) -> None:
    try:
        comments = collect_github_data(github_token, repo_id)
    except (ConnectionError, NameResolutionError):
        logging.error("A networking error occured, aborting")
        exit(1)

    evaluations: Set[CommentEvaluation] = set(convert_comments(comments))

    rprint(f"Evaluating {len(evaluations)} comments")
    # evaluate each comment
    comments_to_process = list(evaluations)[:limit]
    processed_comments = Parallel(n_jobs=-2)(
        delayed(evaluate_comment)(comment, prompt) for comment in tqdm(comments_to_process)
    )

    # second pass, only get the top problematic comments
    concerning_comments = [
        c
        for c in processed_comments
        if c.classification in set([Classification.CONCERNING, Classification.RISK])
    ]
    console.print(f"Running final pass evaluating {len(concerning_comments)} comments")
    final_pass_comments = final_pass(set(concerning_comments))

    print_results(final_pass_comments)
    print_scoreboard(final_pass_comments)
    if excel_output:
        print_results_to_excel(final_pass_comments, excel_output)


def print_results_to_excel(evaluations: List[CommentEvaluation], path: str) -> None:
    df = pd.DataFrame(evaluations)
    # ensure the folder exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, index=False)


def print_scoreboard(evaluations: List[CommentEvaluation]) -> None:
    scoreboard = {}
    for c in evaluations:
        if c.author not in scoreboard:
            scoreboard[c.author] = 0
        scoreboard[c.author] += 1
    scoreboard = sorted(scoreboard.items(), key=lambda x: x[1], reverse=True)
    table = Table(title="Scoreboard")
    table.add_column("Author")
    table.add_column("Count")
    for author, count in scoreboard:
        table.add_row(author, str(count))
    console.print(table)


def print_results(evaluations: List[CommentEvaluation]) -> None:
    table = Table(title="Results")
    table.add_column("Classification")
    table.add_column("Author")
    table.add_column("Content")
    table.add_column("Link")

    # Create a priority mapping for sorting
    priority = {
        Classification.RISK: 0,
        Classification.CONCERNING: 1,
        Classification.UNCLASSIFIED: 2,
        Classification.NEUTRAL: 3,
    }

    # Convert set to list and sort by classification priority
    sorted_evaluations = sorted(
        [ev for ev in evaluations if ev.classification is not None],
        key=lambda x: priority[x.classification],  # type: ignore
    )

    for ev in sorted_evaluations:
        if ev.classification != Classification.NEUTRAL:
            table.add_row(ev.classification, ev.author, ev.extract, ev.link)

    console.print(table)


def convert_comments(comments: List[IssueComment]) -> List[CommentEvaluation]:
    return [
        CommentEvaluation(content=c.body, link=c.html_url, author=c.user.login) for c in comments
    ]


def collect_github_data(github_token: str, repo_id: str) -> List[IssueComment]:
    # loop over all issues
    # get all comments of each issues
    gh = Github(github_token)
    repo = gh.get_repo(repo_id)
    issues = fetch_issues_for_repo(repo_id, gh)
    comments: List[str] = []

    # grab the first posts  of issues
    for issue in issues:
        comments.append(issue.body)

    rprint(f"Fetching all comments a total of {len(issues)} issues")
    responses = Parallel(n_jobs=-2)(
        delayed(fetch_comments_for_issue)(issue.number, repo) for issue in tqdm(issues)
    )
    comments_list: List[IssueComment] = [comment for response in responses for comment in response]
    return comments_list


@memory.cache(ignore=["github"])  # type: ignore
def fetch_issues_for_repo(repo_id: str, github: Github) -> List[Issue]:
    # reinitiate repo object here to enable caching
    repo = github.get_repo(repo_id)
    return list(repo.get_issues(state="all"))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
@memory.cache(ignore=["repo"])  # type: ignore
def fetch_comments_for_issue(issue_id: int, repo: Repository) -> List[IssueComment]:
    comments: List[IssueComment] = []
    issue = repo.get_issue(issue_id)
    if issue.pull_request:
        pr = issue.as_pull_request()
        comments.extend(list(pr.get_review_comments()))  # type: ignore
        comments.extend(list(pr.get_issue_comments()))
    else:
        comments.extend(list(issue.get_comments()))
    return comments


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
@memory.cache(ignore=["model"])  # type: ignore
def evaluate_comment(
    comment: CommentEvaluation, prompt: str, model: str = cli_settings.base_model
) -> CommentEvaluation:
    input_ = comment.content
    generate_content_config = types.GenerateContentConfig(
        temperature=0.15,
        top_p=0.95,
        top_k=40,
        max_output_tokens=1000,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["classification", "extract"],
            properties={
                "classification": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    enum=[x.value for x in Classification if x != Classification.UNCLASSIFIED],
                ),
                "extract": genai.types.Schema(
                    type=genai.types.Type.STRING,
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text=prompt),
        ],
    )

    response = gemini.models.generate_content(
        model=model,
        contents=input_,
        config=generate_content_config,
    ).text

    # try parsing the response as json
    try:
        response = json.loads(response)
        comment.classification = Classification(
            response.get("classification", Classification.UNCLASSIFIED)
        )
        comment.extract = response.get("extract", "")
    except json.JSONDecodeError:
        logging.error(f"Failed to parse response as json: {response}")
        comment.classification = Classification.UNCLASSIFIED

    return comment


@memory.cache()  # type: ignore
def final_pass(comments: Set[CommentEvaluation]):
    input_ = "\n".join([c.short_str() for c in list(comments)])
    generate_content_config = types.GenerateContentConfig(
        temperature=0.15,
        top_p=0.95,
        top_k=40,
        max_output_tokens=65_000,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            properties={
                "comments": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "link": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "classification": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                enum=[
                                    x.value
                                    for x in Classification
                                    if x != Classification.UNCLASSIFIED
                                    and x != Classification.NEUTRAL
                                ],
                            ),
                        },
                        required=["link", "classification"],
                    ),
                ),
            },
            required=["comments"],
        ),
        system_instruction=[
            types.Part.from_text(text=comment_settings.second_pass_prompt),
        ],
    )

    response = gemini.models.generate_content(
        model=comment_settings.advanced_model,
        contents=input_,
        config=generate_content_config,
    )
    # quick lookup dict to map link to comment
    link_to_comment = {c.link: c for c in comments}

    # truly problematic comments
    truly_problematic: List[CommentEvaluation] = []

    # try parsing the response as json
    try:
        res = json.loads(response.text)
        for r in res.get("comments", []):
            comment = link_to_comment[r.get("link")]
            comment.classification = Classification(
                r.get("classification", Classification.UNCLASSIFIED)
            )
            truly_problematic.append(comment)
    except json.JSONDecodeError:
        logging.error(f"Failed to parse response as json: {response}")
    except Exception as e:
        logging.error(f"Failed to parse response: {e}")
        logging.error(f"Response: {response}")
        raise e

    return truly_problematic
