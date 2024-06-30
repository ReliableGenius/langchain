import json
import random
import time
from typing import List

import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field


class BraveSearchWrapper(BaseModel):
    """Wrapper around the Brave search engine."""

    api_key: str
    """The API key to use for the Brave search engine."""
    search_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the search request."""
    base_url: str = "https://api.search.brave.com/res/v1/web/search"
    """The base URL for the Brave search engine."""

    def run(self, query: str) -> str:
        """Query the Brave search engine and return the results as a JSON string.

        Args:
            query: The query to search for.

        Returns: The results as a JSON string.

        """
        web_search_results = self._search_request(query=query)
        final_results = [
            {
                "title": item.get("title"),
                "link": item.get("url"),
                "snippet": item.get("description"),
            }
            for item in web_search_results
        ]
        return json.dumps(final_results)

    def download_documents(self, query: str) -> List[Document]:
        """Query the Brave search engine and return the results as a list of Documents.

        Args:
            query: The query to search for.

        Returns: The results as a list of Documents.

        """
        results = self._search_request(query)
        return [
            Document(
                page_content=item.get("description"),  # type: ignore[arg-type]
                metadata={"title": item.get("title"), "link": item.get("url")},
            )
            for item in results
        ]
    
    def calculate_exponential_backoff_with_jitter(self, attempt, base=2, cap=60):
        """
        Calculate the wait time with exponential backoff and jitter.

        :param attempt: The current retry attempt number.
        :param base: The base wait time in seconds for the exponential backoff. Default is 2.
        :param cap: The maximum wait time in seconds. Default is 60.
        :return: The calculated wait time with jitter.
        """
        # Exponential backoff calculation
        wait_time = min(cap, base * (2 ** attempt))
        # Adding jitter by randomizing the wait time
        jitter = wait_time / 2 + random.uniform(0, wait_time / 2)
        return jitter
    
    def adjust_request_timing(self, remaining_requests, reset_time):
        """
        Adjust the timing of requests based on remaining quota and reset time.

        :param remaining_requests: The number of requests remaining in the current rate limit window.
        :param reset_time: The time in seconds until the rate limit window resets.
        :return: The recommended wait time in seconds before making the next request.
        """
        if remaining_requests <= 0:
            # If no remaining requests, wait until the reset time.
            return reset_time
        else:
            # Spread out the remaining requests evenly over the reset time.
            return max(1, reset_time / remaining_requests)
        
    def _make_request(self, query: str) -> requests.Response:
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        req = requests.PreparedRequest()
        params = {**self.search_kwargs, **{"q": query}}
        req.prepare_url(self.base_url, params)
        if req.url is None:
            raise ValueError("prepared url is None, this should not happen")

        response = requests.get(req.url, headers=headers)
        return response

    def _search_request(self, query: str, max_retries=3) -> List[dict]:
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            response = self._make_request(query)
            if response.ok:
                return response.json().get("web", {}).get("results", [])
            elif response.status_code == 429:
                remaining = int(response.headers.get("X-RateLimit-Remaining", "1,0").split(",")[0])
                reset_time = int(response.headers.get("X-RateLimit-Reset", "1,0").split(",")[0])
                if remaining == 0:
                    wait_time = self.calculate_exponential_backoff_with_jitter(attempt)
                else:
                    wait_time = self.adjust_request_timing(remaining, reset_time)
                time.sleep(wait_time)
                # The request will be retried in the next iteration of the while loop.
            else:
                raise Exception(f"HTTP error {response.status_code}: {response.reason}")
        raise Exception("Max retries reached")