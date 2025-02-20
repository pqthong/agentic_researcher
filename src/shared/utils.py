"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
"""

from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    print(f"Loading chat model: {fully_specified_name}")
    
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
        print(f"Provider: {provider}, Model: {model}")
        if provider.lower() in ["ollama", ""]:
            print(f"Using ChatOllama with model: {model}")
            return ChatOllama(model=model, temperature=0)
        if provider.lower() in ["google", ""]:
            print(f"Using Google with model: {model}")
            return ChatGoogleGenerativeAI(model=model, temperature=0)
    else:
        provider = ""
        model = fully_specified_name
        print(f"Default provider, Model: {model}")
    
    chat_model = init_chat_model(model, model_provider=provider)
    print(f"Initialized chat model: {chat_model}")
    return chat_model
