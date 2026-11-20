"""Google Gemini LLM wrapper using ChatGoogleGenerativeAI."""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.exceptions import LLMError
from app.config import Settings

logger = logging.getLogger("app")


class LLMService:
    def __init__(self, settings: Settings) -> None:
        if not settings.GOOGLE_API_KEY:
            raise LLMError("GOOGLE_API_KEY is not configured")
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    def generate(self, prompt: str) -> str:
        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", None)
            return content if isinstance(content, str) else str(response)
        except Exception as exc:
            logger.exception("LLM generation failed")
            raise LLMError(str(exc))
