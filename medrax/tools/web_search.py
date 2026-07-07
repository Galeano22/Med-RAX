"""Web search tool for looking up medical imaging guidelines / exam recommendations.

Lets the agent search the live web for follow-up and appropriateness guidance
(ACR Appropriateness Criteria, Fleischner Society nodule follow-up, Lung-RADS,
society guidelines, etc.) given the findings it has established from the image.

Backends, in order of preference:
  1. Tavily  — if TAVILY_API_KEY is set (cleaner, ranked snippets)
  2. DuckDuckGo — no API key required (default fallback)
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Type

import requests
from pydantic import BaseModel, Field

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class WebSearchInput(BaseModel):
    """Input for the medical web search tool."""

    query: str = Field(
        ...,
        description=(
            "A focused search query, ideally naming the finding and the guideline you "
            "want, e.g. 'Fleischner Society follow-up 8mm solid pulmonary nodule' or "
            "'ACR Appropriateness Criteria pleural effusion imaging'."
        ),
    )
    max_results: int = Field(5, description="Number of results to return (1-10).")


class WebSearchTool(BaseTool):
    """Tool that searches the web for imaging guidelines and exam recommendations.

    Use this AFTER establishing the findings (from the classifier, report generator,
    VQA, etc.) to look up evidence-based next-step imaging or follow-up recommendations.
    It returns titles, URLs, and text snippets; the agent should synthesize these into
    a recommendation and cite the sources rather than presenting them verbatim.
    """

    name: str = "web_search"
    description: str = (
        "Search the live web for medical imaging guidelines and exam / follow-up "
        "recommendations based on the current findings. Ideal for questions like "
        "'what follow-up imaging is recommended for this nodule?' or 'which exam is "
        "appropriate next?'. Prefers authoritative sources (ACR Appropriateness "
        "Criteria, Fleischner Society, Lung-RADS, radiology societies). Input is a "
        "search query and optional max_results. Output is a list of {title, url, "
        "snippet}. Synthesize and cite these; do not treat a single result as definitive."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    # Bias generic queries toward clinical/guideline sources without overriding a
    # query that already names its target.
    _CLINICAL_HINT = " clinical imaging guideline recommendation"

    def _tavily(self, query: str, max_results: int) -> List[Dict]:
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            return []
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": key,
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in resp.json().get("results", [])
        ]

    def _duckduckgo(self, query: str, max_results: int) -> List[Dict]:
        # Prefer the maintained `ddgs` package; fall back to the older name.
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        last_err = None
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    hits = ddgs.text(query, max_results=max_results) or []
                return [
                    {"title": h.get("title", ""), "url": h.get("href", ""),
                     "snippet": h.get("body", "")}
                    for h in hits
                ]
            except Exception as e:  # ratelimits / transient network errors
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(
            f"DuckDuckGo unavailable ({last_err}). Set TAVILY_API_KEY for a reliable backend."
        )

    def _run(
        self,
        query: str,
        max_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict, Dict]:
        """Run a web search and return ranked results.

        Returns:
            Tuple[Dict, Dict]: ({"query", "results": [{title, url, snippet}]}, metadata)
        """
        query = query.strip().strip("\"'").strip()
        max_results = max(1, min(int(max_results), 10))
        try:
            results = self._tavily(query, max_results)
            backend = "tavily"
            if not results:
                results = self._duckduckgo(query, max_results)
                backend = "duckduckgo"

            if not results:
                return {"error": "No results found."}, {
                    "query": query,
                    "backend": backend,
                    "status": "empty",
                }

            return (
                {"query": query, "results": results},
                {
                    "query": query,
                    "backend": backend,
                    "num_results": len(results),
                    "status": "completed",
                    "note": (
                        "Web snippets are not verified. Synthesize a recommendation and "
                        "cite the source URLs; corroborate with established guidelines."
                    ),
                },
            )
        except Exception as e:
            return {"error": str(e)}, {"query": query, "status": "failed"}

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict, Dict]:
        """Async wrapper; the underlying HTTP call is synchronous."""
        return self._run(query, max_results)
