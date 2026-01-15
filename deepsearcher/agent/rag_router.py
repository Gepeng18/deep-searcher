from typing import List, Optional, Tuple

from deepsearcher.agent import RAGAgent
from deepsearcher.llm.base import BaseLLM
from deepsearcher.utils import log
from deepsearcher.vector_db import RetrievalResult

# RAG路由器提示词
# 给定代理索引列表和相应的描述，每个代理都有特定的功能。给定一个查询，选择最适合处理该查询的一个代理，并返回索引而不提供其他信息。
RAG_ROUTER_PROMPT = """Given a list of agent indexes and corresponding descriptions, each agent has a specific function.
Given a query, select only one agent that best matches the agent handling the query, and return the index without any other information.

## Question
{query}

## Agent Indexes and Descriptions
{description_str}

Only return one agent index number that best matches the agent handling the query:
"""


# 将查询路由到最合适RAG代理实现的类
class RAGRouter(RAGAgent):
    """
    Routes queries to the most appropriate RAG agent implementation.

    This class analyzes the content and requirements of a query and determines
    which RAG agent implementation is best suited to handle it.
    """

    # 初始化RAGRouter
    def __init__(
        self,
        llm: BaseLLM,
        rag_agents: List[RAGAgent],
        agent_descriptions: Optional[List[str]] = None,
    ):
        """
        Initialize the RAGRouter.

        Args:
            llm: The language model to use for analyzing queries.
            rag_agents: A list of RAGAgent instances.
            agent_descriptions (list, optional): A list of descriptions for each agent.
        """
        self.llm = llm
        self.rag_agents = rag_agents
        self.agent_descriptions = agent_descriptions
        if not self.agent_descriptions:
            try:
                self.agent_descriptions = [
                    agent.__class__.__description__ for agent in self.rag_agents
                ]
            except Exception:
                raise AttributeError(
                    "Please provide agent descriptions or set __description__ attribute for each agent class."
                )

    # 路由查询到最合适的代理
    def _route(self, query: str) -> Tuple[RAGAgent, int]:
        description_str = "\n".join(
            [f"[{i + 1}]: {description}" for i, description in enumerate(self.agent_descriptions)]
        )
        prompt = RAG_ROUTER_PROMPT.format(query=query, description_str=description_str)
        chat_response = self.llm.chat(messages=[{"role": "user", "content": prompt}])
        try:
            selected_agent_index = int(self.llm.remove_think(chat_response.content)) - 1
        except ValueError:
            # 在某些推理LLM中，输出不是数字，而是一个解释字符串，最后带有数字。
            # In some reasoning LLM, the output is not a number, but a explaination string with a number in the end.
            log.warning(
                "Parse int failed in RAGRouter, but will try to find the last digit as fallback."
            )
            selected_agent_index = (
                int(self.find_last_digit(self.llm.remove_think(chat_response.content))) - 1
            )

        selected_agent = self.rag_agents[selected_agent_index]
        log.color_print(
            f"<think> Select agent [{selected_agent.__class__.__name__}] to answer the query [{query}] </think>\n"
        )
        return self.rag_agents[selected_agent_index], chat_response.total_tokens

    def retrieve(self, query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        agent, n_token_router = self._route(query)
        retrieved_results, n_token_retrieval, metadata = agent.retrieve(query, **kwargs)
        return retrieved_results, n_token_router + n_token_retrieval, metadata

    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        agent, n_token_router = self._route(query)
        answer, retrieved_results, n_token_retrieval = agent.query(query, **kwargs)
        return answer, retrieved_results, n_token_router + n_token_retrieval

    # 查找字符串中的最后一个数字
    def find_last_digit(self, string):
        for char in reversed(string):
            if char.isdigit():
                return char
        raise ValueError("No digit found in the string")
