from typing import List, Tuple

from deepsearcher.agent.base import BaseAgent
from deepsearcher.llm.base import BaseLLM
from deepsearcher.utils import log
from deepsearcher.vector_db.base import BaseVectorDB

# 集合路由提示词
# 我为你提供集合名称和相应的集合描述。请选择可能与问题相关的集合名称，并返回一个Python字符串列表。如果没有与问题相关的集合，你可以返回一个空列表。
COLLECTION_ROUTE_PROMPT = """
I provide you with collection_name(s) and corresponding collection_description(s). Please select the collection names that may be related to the question and return a python list of str. If there is no collection related to the question, you can return an empty list.

"QUESTION": {question}
"COLLECTION_INFO": {collection_info}

When you return, you can ONLY return a python list of str, WITHOUT any other additional content. Your selected collection name list is:
"""


# 将查询路由到向量数据库中适当集合的类
class CollectionRouter(BaseAgent):
    """
    Routes queries to appropriate collections in the vector database.

    This class analyzes the content of a query and determines which collections
    in the vector database are most likely to contain relevant information.
    """

    # 初始化CollectionRouter
    def __init__(self, llm: BaseLLM, vector_db: BaseVectorDB, dim: int, **kwargs):
        """
        Initialize the CollectionRouter.

        Args:
            llm: The language model to use for analyzing queries.
            vector_db: The vector database containing the collections.
            dim: The dimension of the vector space to search in.
        """
        self.llm = llm
        self.vector_db = vector_db
        self.all_collections = [
            collection_info.collection_name
            for collection_info in self.vector_db.list_collections(dim=dim)
        ]

    # 确定给定查询的相关集合
    def invoke(self, query: str, dim: int, **kwargs) -> Tuple[List[str], int]:
        """
        Determine which collections are relevant for the given query.

        This method analyzes the query content and selects collections that are
        most likely to contain information relevant to answering the query.

        Args:
            query (str): The query to analyze.
            dim (int): The dimension of the vector space to search in.

        Returns:
            Tuple[List[str], int]: A tuple containing:
                - A list of selected collection names
                - The token usage for the routing operation
        """
        # 初始化token消耗计数器
        consume_tokens = 0
        # 获取指定维度下的所有集合信息
        collection_infos = self.vector_db.list_collections(dim=dim)
        # 检查向量数据库中是否存在集合
        if len(collection_infos) == 0:
            log.warning(
                "No collections found in the vector database. Please check the database connection."
            )
            return [], 0
        # 如果只有一个集合，则直接使用这个集合进行搜索
        if len(collection_infos) == 1:
            # 获取唯一的集合名称
            the_only_collection = collection_infos[0].collection_name
            # 记录搜索信息到日志
            log.color_print(
                f"<think> Perform search [{query}] on the vector DB collection: {the_only_collection} </think>\n"
            )
            # 返回唯一的集合名称和0token消耗（因为没有使用LLM进行路由选择）
            return [the_only_collection], 0
        # 构建向量数据库搜索提示词，用于让LLM选择相关的集合
        vector_db_search_prompt = COLLECTION_ROUTE_PROMPT.format(
            question=query,  # 用户查询问题
            collection_info=[  # 所有可用集合的信息列表
                {
                    "collection_name": collection_info.collection_name,  # 集合名称
                    "collection_description": collection_info.description,  # 集合描述
                }
                for collection_info in collection_infos  # 遍历所有集合信息
            ],
        )
        # 调用LLM来分析查询并选择相关的集合
        chat_response = self.llm.chat(
            messages=[{"role": "user", "content": vector_db_search_prompt}]
        )
        # 解析LLM返回的集合名称列表
        selected_collections = self.llm.literal_eval(chat_response.content)
        # 累加LLM调用的token消耗
        consume_tokens += chat_response.total_tokens

        # 遍历所有集合信息，添加额外的集合选择逻辑
        for collection_info in collection_infos:
            # 如果未提供集合描述，则将查询用作搜索查询
            # If a collection description is not provided, use the query as the search query
            if not collection_info.description:
                selected_collections.append(collection_info.collection_name)
            # 如果存在默认集合，则将查询用作搜索查询
            # If the default collection exists, use the query as the search query
            # 如果当前集合是默认集合，则确保它被包含在搜索中
            if self.vector_db.default_collection == collection_info.collection_name:
                selected_collections.append(collection_info.collection_name)
        # 去除重复的集合名称并转换为列表
        selected_collections = list(set(selected_collections))
        # 记录最终选择的集合信息到日志
        log.color_print(
            f"<think> Perform search [{query}] on the vector DB collections: {selected_collections} </think>\n"
        )
        # 返回选择的集合列表和总token消耗
        return selected_collections, consume_tokens
