from typing import List, Tuple

# 从deepsearcher模块导入配置，全局变量通过configuration模块访问
from deepsearcher import configuration
from deepsearcher.vector_db.base import RetrievalResult


# 使用问题查询知识库并获取答案
def query(original_query: str, max_iter: int = 3) -> Tuple[str, List[RetrievalResult], int]:
    """
    使用问题查询知识库并获取答案。

    此函数使用默认搜索器查询知识库，并基于检索到的信息生成答案。
    这是DeepSearcher系统的主要查询接口，支持复杂的RAG检索和答案生成。

    参数:
        original_query: 要搜索的问题或查询。
        max_iter: 搜索过程的最大迭代次数。默认为3次，用于控制搜索的深度。

    返回:
        包含以下元素的元组:
            - 生成的答案（字符串格式）
            - 用于生成答案的检索结果列表
            - 过程中消耗的token数量
    """
    # 获取配置的默认搜索器实例（RAGRouter）
    default_searcher = configuration.default_searcher
    # 调用搜索器的query方法执行查询并返回结果
    return default_searcher.query(original_query, max_iter=max_iter)


# 从知识库检索相关信息而不生成答案
def retrieve(
    original_query: str, max_iter: int = 3
) -> Tuple[List[RetrievalResult], List[str], int]:
    """
    从知识库检索相关信息而不生成答案。

    此函数使用默认搜索器从知识库中检索与查询相关的信息，
    只返回检索结果，不进行答案生成。适用于需要原始检索数据的场景。

    参数:
        original_query: 要搜索的问题或查询。
        max_iter: 搜索过程的最大迭代次数。

    返回:
        包含以下元素的元组:
            - 检索结果列表
            - 空列表（为将来使用保留的占位符）
            - 过程中消耗的token数量
    """
    # 获取配置的默认搜索器实例
    default_searcher = configuration.default_searcher
    # 调用搜索器的retrieve方法执行检索，返回检索结果、消耗的token和元数据
    retrieved_results, consume_tokens, metadata = default_searcher.retrieve(
        original_query, max_iter=max_iter
    )
    # 返回检索结果、空列表和消耗的token数量
    return retrieved_results, [], consume_tokens


# 使用朴素RAG方法进行简单的知识库检索
def naive_retrieve(query: str, collection: str = None, top_k=10) -> List[RetrievalResult]:
    """
    使用朴素RAG方法进行简单的知识库检索。

    此函数使用朴素RAG代理从知识库中检索信息，
    不使用任何高级技术如迭代精炼。适合需要快速、简单检索的场景。

    参数:
        query: 要搜索的问题或查询。
        collection: 要搜索的集合名称。如果为None，则在所有集合中搜索。
        top_k: 返回的最大结果数量。默认为10。

    返回:
        检索结果列表。
    """
    # 获取配置的朴素RAG实例
    naive_rag = configuration.naive_rag
    # 调用朴素RAG的retrieve方法执行检索，返回所有检索结果、消耗的token和元数据
    all_retrieved_results, consume_tokens, _ = naive_rag.retrieve(query)
    # 返回所有检索结果
    return all_retrieved_results


# 使用朴素RAG方法查询知识库并获取答案
def naive_rag_query(
    query: str, collection: str = None, top_k=10
) -> Tuple[str, List[RetrievalResult]]:
    """
    使用朴素RAG方法查询知识库并获取答案。

    此函数使用朴素RAG代理查询知识库并基于检索到的信息生成答案，
    不使用任何高级技术。适合需要快速答案生成的简单场景。

    参数:
        query: 要搜索的问题或查询。
        collection: 要搜索的集合名称。如果为None，则在所有集合中搜索。
        top_k: 要考虑的最大结果数量。默认为10。

    返回:
        包含以下元素的元组:
            - 生成的答案（字符串格式）
            - 用于生成答案的检索结果列表
    """
    # 获取配置的朴素RAG实例
    naive_rag = configuration.naive_rag
    # 调用朴素RAG的query方法执行查询并生成答案，返回答案、检索结果和消耗的token
    answer, retrieved_results, consume_tokens = naive_rag.query(query)
    # 返回答案和检索结果
    return answer, retrieved_results
