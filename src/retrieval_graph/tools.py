"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""
from index_graph.graph import graph as index_graph

from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from shared.configuration import BaseConfiguration as Configuration


async def web_search_tool(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=10)
    result = await wrapped.ainvoke({"query": query})
    result = cast(list[dict[str, Any]], result)
    print(f"search    {result}")
    # if result:
    #         # Send the search results to the index graph
    #         await index_graph.invoke({"documents": result})
    return result


TOOLS: List[Callable[..., Any]] = [web_search_tool]

