"""Perplexity search utility module with academic and web search support."""

import os
import requests
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class PerplexitySearch:
    """Wrapper for Perplexity API search functionality with academic and web search modes."""
    
    def __init__(self, model: str = "sonar"):
        """
        Initialize Perplexity search client.
        
        Args:
            model: Perplexity model to use (default: "sonar")
            
        Raises:
            ValueError: If PERPLEXITY_API_KEY is not set
        """
        self.api_key = os.environ.get("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable is not set")
            
        self.url = "https://api.perplexity.ai/chat/completions"
        self.model = model
    
    def _make_request(
        self,
        query: str,
        search_mode: Optional[str] = None,
        search_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_recency_filter: Optional[str] = None,
        search_after_date: Optional[str] = None,
        search_before_date: Optional[str] = None,
        search_context_size: str = "high",
        system_prompt: str = "Be precise and concise. Provide citations for all facts.",
        max_tokens: int = 4096,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Make a request to Perplexity API with specified parameters.
        
        Args:
            query: Search query string
            search_mode: Search mode ("academic" for academic sources, None for general web)
            search_domains: List of domains to include in search
            exclude_domains: List of domains to exclude from search
            search_recency_filter: Time filter ('day', 'week', 'month', 'year')
            search_after_date: Include content after this date (YYYY-MM-DD)
            search_before_date: Include content before this date (YYYY-MM-DD)
            search_context_size: Context size ('low', 'medium', 'high')
            system_prompt: System prompt for the model
            max_tokens: Maximum tokens in response
            temperature: Model temperature
            
        Returns:
            API response as dictionary
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "return_images": False,
            "return_related_questions": False,
            "top_k": 5,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1,
            "web_search_options": {"search_context_size": search_context_size}
        }
        
        # Add search mode if specified
        if search_mode:
            payload["search_mode"] = search_mode
        
        # Add domain filters
        if search_domains or exclude_domains:
            domains = []
            if search_domains:
                domains.extend(search_domains)
            if exclude_domains:
                domains.extend([f"-{domain}" for domain in exclude_domains])
            if domains:
                payload["search_domain_filter"] = domains[:10]  # API limit of 10 domains
        
        # Add time filters
        if search_recency_filter:
            payload["search_recency_filter"] = search_recency_filter
        if search_after_date:
            payload["search_after_date_filter"] = search_after_date
        if search_before_date:
            payload["search_before_date_filter"] = search_before_date
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Perplexity API request failed: {str(e)}")
    
    def _format_response_with_citations(self, response_data: Dict[str, Any]) -> str:
        """
        Format API response with properly numbered citations.
        
        Args:
            response_data: Raw API response
            
        Returns:
            Formatted response with citations
        """
        if 'choices' not in response_data or not response_data['choices']:
            raise ValueError("Invalid response format from Perplexity API")
        
        answer = response_data['choices'][0]['message']['content']
        
        # Extract citations if available
        citations = response_data.get('search_results', [])
        
        # Format citations with proper numbering
        if citations:
            formatted_citations = [f"[{idx+1}] {citation['title']} {citation['url']}" for idx, citation in enumerate(citations)]
            return f"{answer}\n\n## Citations\n" + "\n".join(formatted_citations)
        
        return answer
    
    def academic_search(
        self,
        query: str,
        search_after_date: Optional[str] = None,
        search_before_date: Optional[str] = None,
        max_tokens: int = 4096
    ) -> str:
        """
        Perform an academic search focusing on scholarly sources.
        
        Args:
            query: Academic search query
            search_after_date: Include papers after this date (YYYY-MM-DD)
            search_before_date: Include papers before this date (YYYY-MM-DD)
            max_tokens: Maximum tokens in response
            
        Returns:
            Academic search results with citations
        """
        system_prompt = (
            "You are an academic research assistant. Provide comprehensive answers based on "
            "peer-reviewed papers, academic journals, and scholarly sources. "
            "Always cite your sources with proper academic citations."
        )
        
        response_data = self._make_request(
            query=query,
            search_mode="academic",
            search_after_date=search_after_date,
            search_before_date=search_before_date,
            search_context_size="high",
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.1  # Lower temperature for academic precision
        )
        
        return self._format_response_with_citations(response_data)
    
    def web_search(
        self,
        query: str,
        search_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_recency_filter: Optional[str] = None,
        search_context_size: str = "high",
        max_tokens: int = 4096
    ) -> str:
        """
        Perform a general web search with optional domain and recency filters.
        
        Args:
            query: Web search query
            search_domains: List of domains to include (max 10)
            exclude_domains: List of domains to exclude
            search_recency_filter: Time filter ('day', 'week', 'month', 'year')
            search_context_size: Context size ('low', 'medium', 'high')
            max_tokens: Maximum tokens in response
            
        Returns:
            Web search results with citations
        """
        system_prompt = (
            "Provide accurate and comprehensive answers based on current web information. "
            "Always cite your sources with numbered references."
        )
        
        response_data = self._make_request(
            query=query,
            search_domains=search_domains,
            exclude_domains=exclude_domains,
            search_recency_filter=search_recency_filter,
            search_context_size=search_context_size,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )
        
        return self._format_response_with_citations(response_data)
    
    def search(self, query: str, **kwargs) -> str:
        """
        Perform a search with custom parameters (backward compatible).
        
        Args:
            query: Search query string
            **kwargs: Additional parameters to pass to _make_request
            
        Returns:
            Search results with citations as formatted string
        """
        response_data = self._make_request(query, **kwargs)
        return self._format_response_with_citations(response_data)


# Convenience functions for easy importing
def academic_search(
    query: str,
    model: str = "sonar",
    search_after_date: Optional[str] = None,
    search_before_date: Optional[str] = None,
    max_tokens: int = 4096
) -> str:
    """
    Convenience function for academic search using Perplexity.
    
    Args:
        query: Academic search query
        model: Perplexity model to use (default: "sonar")
        search_after_date: Include papers after this date (YYYY-MM-DD)
        search_before_date: Include papers before this date (YYYY-MM-DD)
        max_tokens: Maximum tokens in response
        
    Returns:
        Academic search results with citations
        
    Example:
        >>> result = academic_search("quantum computing applications in cryptography")
        >>> print(result)
    """
    searcher = PerplexitySearch(model=model)
    return searcher.academic_search(
        query=query,
        search_after_date=search_after_date,
        search_before_date=search_before_date,
        max_tokens=max_tokens
    )


def general_web_search(
    query: str,
    model: str = "sonar-deep-research",
    search_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    search_recency_filter: Optional[str] = None,
    search_context_size: str = "high",
    max_tokens: int = 4096
) -> str:
    """
    Convenience function for general web search using Perplexity.
    
    Args:
        query: Web search query
        model: Perplexity model to use (default: "sonar")
        search_domains: List of domains to include (max 10)
        exclude_domains: List of domains to exclude
        search_recency_filter: Time filter ('day', 'week', 'month', 'year')
        search_context_size: Context size ('low', 'medium', 'high')
        max_tokens: Maximum tokens in response
        
    Returns:
        Web search results with citations
        
    Example:
        >>> result = general_web_search(
        ...     "latest AI developments",
        ...     search_recency_filter="week",
        ...     exclude_domains=["example.com"]
        ... )
        >>> print(result)
    """
    searcher = PerplexitySearch(model=model)
    return searcher.web_search(
        query=query,
        search_domains=search_domains,
        exclude_domains=exclude_domains,
        search_recency_filter=search_recency_filter,
        search_context_size=search_context_size,
        max_tokens=max_tokens
    )


# Backward compatibility
def search_web(query: str, model: str = "sonar-reasoning-pro") -> str:
    """
    Backward compatible convenience function to search the web using Perplexity.
    
    Args:
        query: Search query string
        model: Perplexity model to use (default: "sonar")
        
    Returns:
        Search results with citations as formatted string
    """
    return general_web_search(query, model=model)
if __name__ == "__main__":
    print(general_web_search("what is the capital of France?"))
    print(academic_search("what main contribution of VGG-16?"))