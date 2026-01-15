from abc import ABC
from typing import Any, List, Tuple

from deepsearcher.vector_db import RetrievalResult


# 为类添加描述的装饰器函数
def describe_class(description):
    """
    Decorator function to add a description to a class.

    This decorator adds a __description__ attribute to the decorated class,
    which can be used for documentation or introspection.

    Args:
        description: The description to add to the class.

    Returns:
        A decorator function that adds the description to the class.
    """

    def decorator(cls):
        cls.__description__ = description
        return cls

    return decorator


# DeepSearcher系统中所有代理的抽象基类
class BaseAgent(ABC):
    """
    Abstract base class for all agents in the DeepSearcher system.

    This class defines the basic interface for agents, including initialization
    and invocation methods.
    """

    # 初始化BaseAgent对象
    def __init__(self, **kwargs):
        """
        Initialize a BaseAgent object.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    # 调用代理并返回结果
    def invoke(self, query: str, **kwargs) -> Any:
        """
        Invoke the agent and return the result.

        Args:
            query: The query string.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of invoking the agent.
        """


# 检索增强生成（RAG）代理的抽象基类
class RAGAgent(BaseAgent):
    """
    Abstract base class for Retrieval-Augmented Generation (RAG) agents.

    This class extends BaseAgent with methods specific to RAG, including
    retrieval and query methods.
    """

    # 初始化RAGAgent对象
    def __init__(self, **kwargs):
        """
        Initialize a RAGAgent object.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    # 从知识库检索文档结果
    def retrieve(self, query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """
        Retrieve document results from the knowledge base.

        Args:
            query: The query string.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple containing:
                - the retrieved results
                - the total number of token usages of the LLM
                - any additional metadata, which can be an empty dictionary
        """

    # 查询代理并返回答案
    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        Query the agent and return the answer.

        Args:
            query: The query string.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple containing:
                - the result generated from LLM
                - the retrieved document results
                - the total number of token usages of the LLM
        """
