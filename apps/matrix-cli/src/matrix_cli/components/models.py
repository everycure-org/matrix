from typing import Dict, Optional

from pydantic import BaseModel, Field


class PRInfo(BaseModel):
    """Pydantic model for PR information."""

    number: int
    title: str
    current_labels: str = Field(description="Comma-separated list of current labels")
    new_title: str
    new_labels: str = Field(description="Comma-separated list of new labels")
    head_ref_name: str = Field(default="", description="Name of the branch the PR was created from")
    url: str
    diff: str = ""
    merge_commit: Optional[str] = Field(default=None, description="Git merge commit hash")
    ai_suggested_title: Optional[str] = Field(default=None, description="Title suggested by AI")

    @classmethod
    def from_github_response(cls, pr_info: Dict, diff: str = "") -> "PRInfo":
        """
        Create a PRInfo instance from GitHub API response.

        Args:
            pr_info (Dict): Raw GitHub API response
            diff (str): Git diff content

        Returns:
            PRInfo: Structured PR information
        """
        labels = ",".join([label["name"] for label in pr_info.get("labels", [])])
        return cls(
            number=pr_info["number"],
            title=pr_info["title"],
            current_labels=labels,
            new_title=pr_info["title"],  # Initially same as current title
            new_labels=labels,  # Initially same as current labels
            head_ref_name=pr_info.get("headRefName", ""),
            url=pr_info["url"],
            diff=diff,
            merge_commit=pr_info.get("mergeCommit") and pr_info["mergeCommit"].get("oid", ""),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding the diff to avoid Excel issues."""
        return self.model_dump()
