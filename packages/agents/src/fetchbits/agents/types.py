from typing import Any, TypeVar

from pydantic import BaseModel

from ragbits.agents._main import Agent
from ragbits.core.llms.base import LLMClientOptionsT

QuestionAnswerPromptInputT = TypeVar("QuestionAnswerPromptInputT", bound="QuestionAnswerPromptInput")
QuestionAnswerPromptOutputT = TypeVar("QuestionAnswerPromptOutputT", bound="QuestionAnswerPromptOutput | str")

QuestionAnswerAgent = Agent[LLMClientOptionsT, QuestionAnswerPromptInputT, QuestionAnswerPromptOutputT]

class QuestionAnswerPromptInput(BaseModel):
    question : str
    context : Any | None = None

class QuestionAnswerPromptOutput(BaseModel):
    """
    Output for the question answer agent.
    """

    answer: str
    """The answer to the question."""