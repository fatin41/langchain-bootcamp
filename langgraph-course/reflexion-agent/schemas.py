from typing import List
from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superflous: str = Field(description="Critique of what is superflous.")


class AnswerQuestion(BaseModel):
    answer: str = Field(
        description="250 word detailed answer to the the question.")
    reflection: Reflection = Field(
        description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to the address the critique of your current answer.")
