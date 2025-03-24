#!/bin/python
import base64
import logging
import os

import typer
from github import Github, Issue, IssueComment, Repository
from google import genai
from google.genai import types
from joblib import Memory, Parallel, delayed
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from rich import print as rprint
from tqdm.rich import tqdm

from matrix_cli.components.settings import settings as cli_settings
from matrix_cli.components.utils import invoke_model

app = typer.Typer(help="Github comment-related utility commands", no_args_is_help=True)
logging.basicConfig(level=logging.INFO)
memory = Memory(location=".cache", verbose=0)


class CommentEvaluationSettings(BaseSettings):
    prompt: str = """
    Your objective is the detection of harmful comments on our github as we prepare for an open sourcing of the repository. 
    Please identify each comment as "neutral", "concerning" or "identified risk".

    - Neutral: Comments that are just focused on the topic or are otherwise not noteworthy
    - Concerning: Comments that may be misinterpreted, could be perceived as hostile or are unprofessional
    - Identified Risk: Comments that clearly break common code of conducts or are unfair, mean or otherwise risky to be put online.
    """
    model: str = cli_settings.base_model


@app.command()
def detect_risk(
    ids: str = typer.Option(help="The issues or PRs to evaluate. Comma separated list", default=None),
    github_token: str = typer.Option(help="The Github token to use", envvar="GITHUB_TOKEN"),
    repo_id: str = typer.Option(help="The repo id to use", default="everycure-org/matrix"),
):
    # loop over all issues
    # get all comments of each issues
    gh = Github(github_token)
    repo = gh.get_repo(repo_id)
    issues = fetch_issues_for_repo(repo_id, gh)
    comments = []
    for issue in issues:
        comments.append(issue.body)

    rprint(f"Fetching all comments a total of {len(issues)} issues")
    responses = Parallel(n_jobs=-2)(delayed(fetch_comments_for_issue)(issue.number, repo) for issue in tqdm(issues))
    comments = [comment for response in responses for comment in response]

    rprint(f"Evaluating {len(comments)} comments")
    # evaluate each comment
    for comment in tqdm(comments):
        evaluate_comment(comment.body, prompt)


@memory.cache(ignore=["github"])
def fetch_issues_for_repo(repo_id: str, github: Github) -> list[Issue]:
    # reinitiate repo object here to enable caching
    repo = github.get_repo(repo_id)
    return list(repo.get_issues(state="all"))


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


class CommentEvaluation(BaseModel):
    content: str
    link: str
    author: str
    classification: str
    extract: str


def evaluate_comment(input: str, prompt: str, model: str = cli_settings.base_model) -> CommentEvaluation:
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
                    enum=["neutral", "concerning", "identified_risk"],
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

    response = invoke_model(prompt=input, model=model, generation_config=generate_content_config)
    return response


if __name__ == "__main__":
    generate()
