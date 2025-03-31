#!/bin/python
import logging
import os
from enum import Enum
from typing import Optional

import typer
from dotenv import load_dotenv
from github import Github, Issue, IssueComment, Repository
from google import genai
from google.genai import types
from joblib import Memory, Parallel, delayed
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.rich import tqdm
from urllib3.exceptions import NameResolutionError

from matrix_cli.components.settings import settings as cli_settings
from matrix_cli.components.utils import invoke_model

# Need to ensure this happens before we initialize the genai client
load_dotenv()

app = typer.Typer(help="Github comment-related utility commands", no_args_is_help=True)
logging.basicConfig(level=logging.INFO)
memory = Memory(location=".cache", verbose=0)
console = Console()

gemini = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)


class Classification(str, Enum):
    NEUTRAL = "neutral"
    CONCERNING = "concerning"
    RISK = "risk"


class CommentEvaluation(BaseModel):
    content: str
    link: str
    author: str
    classification: Optional[Classification] = None
    extract: Optional[str] = None

    def __hash__(self):
        return hash((self.link))


class CommentEvaluationSettings(BaseSettings):
    prompt: str = """
    Your objective is the detection of harmful comments on our github as we prepare for
    an open sourcing of the repository.
    Please identify each comment as "neutral", "concerning" or "identified risk".

    - Neutral: Comments that are just focused on the topic or are otherwise not noteworthy
    - Concerning: Comments that may be misinterpreted, could be perceived as hostile or
      are unprofessional
    - Identified Risk: Comments that clearly break common code of conducts or are unfair,
      mean or otherwise risky to be put online.
    """
    model: str = cli_settings.base_model


comment_settings = CommentEvaluationSettings()


@app.command()
def detect_risk(
    ids: str = typer.Option(
        help="The issues or PRs to evaluate. Comma separated list", default=None
    ),
    github_token: str = typer.Option(help="The Github token to use", envvar="GITHUB_TOKEN"),
    repo_id: str = typer.Option(help="The repo id to use", default="everycure-org/matrix"),
    prompt: str = typer.Option(help="The prompt to use", default=comment_settings.prompt),
    limit: int = typer.Option(help="The limit of comments to evaluate", default=None),
):
    try:
        comments = collect_github_data(github_token, repo_id)
    except (ConnectionError, NameResolutionError):
        logging.error("A networking error occured, aborting")
        exit(1)

    evaluations = set(convert_comments(comments))

    rprint(f"Evaluating {len(evaluations)} comments")
    # evaluate each comment
    for comment in tqdm(evaluations[:limit]):
        # replace object with the one containing evaluation
        evaluations.add(evaluate_comment(comment, prompt))
        # evaluations.add(dummy_evaluate_comment(comment, prompt))

    print_results(evaluations)


def print_results(evaluations: set[CommentEvaluation]):
    table = Table(title="Results")
    table.add_column("Classification")
    table.add_column("Author")
    table.add_column("Content")
    table.add_column("Link")

    # Create a priority mapping for sorting
    priority = {
        Classification.RISK: 0,
        Classification.CONCERNING: 1,
        Classification.NEUTRAL: 2,
    }

    # Convert set to list and sort by classification priority
    sorted_evaluations = sorted(evaluations, key=lambda x: priority[x.classification])

    for ev in sorted_evaluations:
        if ev.classification != Classification.NEUTRAL:
            table.add_row(ev.classification, ev.author, ev.extract, ev.link)

    console.print(table)


def convert_comments(comments: list[IssueComment]) -> list[CommentEvaluation]:
    return [CommentEvaluation(content=c.body, link=c.url, author=c.user.login) for c in comments]


def collect_github_data(github_token, repo_id) -> list[IssueComment]:
    # loop over all issues
    # get all comments of each issues
    gh = Github(github_token)
    repo = gh.get_repo(repo_id)
    issues = fetch_issues_for_repo(repo_id, gh)
    comments = []

    # grab the first posts  of issues
    for issue in issues:
        comments.append(issue.body)

    rprint(f"Fetching all comments a total of {len(issues)} issues")
    responses = Parallel(n_jobs=-2)(
        delayed(fetch_comments_for_issue)(issue.number, repo) for issue in tqdm(issues)
    )
    comments = [comment for response in responses for comment in response]
    return comments


@memory.cache(ignore=["github"])
def fetch_issues_for_repo(repo_id: str, github: Github) -> list[Issue]:
    # reinitiate repo object here to enable caching
    repo = github.get_repo(repo_id)
    return list(repo.get_issues(state="all"))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
@memory.cache(ignore=["repo"])
def fetch_comments_for_issue(issue_id: int, repo: Repository) -> list[IssueComment]:
    comments = []
    issue = repo.get_issue(issue_id)
    if issue.pull_request:
        pr = issue.as_pull_request()
        comments.extend(list(pr.get_review_comments()))
        comments.extend(list(pr.get_issue_comments()))
    else:
        comments.extend(list(issue.get_comments()))
    return comments


def dummy_evaluate_comment(
    comment: CommentEvaluation, prompt: str, model: str = cli_settings.base_model
):
    import random

    comment.classification = random.choice([x.value for x in Classification])
    comment.extract = "NOT AN EXTRACT"
    return comment


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
@memory.cache
def evaluate_comment(
    comment: CommentEvaluation, prompt: str, model: str = cli_settings.base_model
) -> CommentEvaluation:
    input_ = comment.content
    generate_content_config = types.GenerateContentConfig(
        temperature=0.15,
        top_p=0.95,
        top_k=40,
        max_output_tokens=1000,
        # by default, gemini will not block unsafe content
        # safety_settings=[types.SafetySetting()],
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["classification", "extract"],
            properties={
                "classification": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    enum=[x.value for x in Classification],
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

    response = invoke_model(prompt=input_, model=model, generation_config=generate_content_config)
    response = gemini.models.generate_content(
        model=model,
        contents=input_,
        config=generate_content_config,
    ).text
    return response
