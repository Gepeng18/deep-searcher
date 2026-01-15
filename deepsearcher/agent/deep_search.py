import asyncio
from typing import List, Tuple

from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.utils import log
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB, deduplicate_results

# 子查询生成提示词
# 为了更全面地回答这个问题，请将原始问题拆解为最多四个子问题。请以字符串列表（list of str）的形式返回。
# 如果这是一个非常简单的问题，无需拆解，则请在 Python 代码列表中仅保留原本的那个问题。
#
# 原始问题：{original_query}
#
#
# <示例>
# 输入示例：
# "Explain deep learning"（解释深度学习）
#
# 输出示例：
# [
#     "What is deep learning?",
#     "What is the difference between deep learning and machine learning?",
#     "What is the history of deep learning?"
# ]
# </示例>
#
# 请以 Python 代码字符串列表的格式提供你的回答：
SUB_QUERY_PROMPT = """To answer this question more comprehensively, please break down the original question into up to four sub-questions. Return as list of str.
If this is a very simple question and no decomposition is necessary, then keep the only one original question in the python code list.

Original Question: {original_query}


<EXAMPLE>
Example input:
"Explain deep learning"

Example output:
[
    "What is deep learning?",
    "What is the difference between deep learning and machine learning?",
    "What is the history of deep learning?"
]
</EXAMPLE>

Provide your response in a python code list of str format:
"""

# 重排序提示词
# 请根据查询问题和检索到的文本片段，判断该片段是否有助于回答任何查询问题。请仅返回 "YES" 或 "NO"，不要包含任何其他信息。
#
# 查询问题：{query}
# 检索到的片段：{retrieved_chunk}
#
# 该片段是否有助于回答上述任何问题？
RERANK_PROMPT = """Based on the query questions and the retrieved chunk, to determine whether the chunk is helpful in answering any of the query question, you can only return "YES" or "NO", without any other information.

Query Questions: {query}
Retrieved Chunk: {retrieved_chunk}

Is the chunk helpful in answering the any of the questions?
"""


# 反思提示词
# 请根据原始查询、此前的子查询以及所有检索到的文档片段，判断是否需要执行额外的搜索查询。
# 如果需要进一步研究，请提供一个包含最多 3 个搜索查询的 Python 列表。
# 如果无需进一步研究，请返回一个空列表 `[]`。
#
# 注意：如果原始查询的要求是撰写报告，请倾向于生成更多后续查询，而非直接返回空列表。
#
# 原始查询：{question}
#
# 此前的子查询：{mini_questions}
#
# 相关文档片段：
# {mini_chunk_str}
#
# 请仅以有效的字符串列表（List of str）格式回答，不要包含任何其他文本。
REFLECT_PROMPT = """Determine whether additional search queries are needed based on the original query, previous sub queries, and all retrieved document chunks. If further research is required, provide a Python list of up to 3 search queries. If no further research is required, return an empty list.

If the original query is to write a report, then you prefer to generate some further queries, instead return an empty list.

Original Query: {question}

Previous Sub Queries: {mini_questions}

Related Chunks: 
{mini_chunk_str}

Respond exclusively in valid List of str format without any other text."""


# 总结提示词
# 你是一位 AI 内容分析专家，擅长内容归纳与总结。请根据此前的查询以及检索到的文档片段，总结出一份具体且详尽的回答或报告。
#
# 原始查询：{question}
#
# 此前的子查询：{mini_questions}
#
# 相关文档片段：
# {mini_chunk_str}
SUMMARY_PROMPT = """You are a AI content analysis expert, good at summarizing content. Please summarize a specific and detailed answer or report based on the previous queries and the retrieved document chunks.

Original Query: {question}

Previous Sub Queries: {mini_questions}

Related Chunks: 
{mini_chunk_str}

"""


# 深度搜索代理实现
@describe_class(
    "This agent is suitable for handling general and simple queries, such as given a topic and then writing a report, survey, or article."
)
class DeepSearch(RAGAgent):
    """
    Deep Search agent implementation for comprehensive information retrieval.

    This agent performs a thorough search through the knowledge base, analyzing
    multiple aspects of the query to provide comprehensive and detailed answers.
    """

    # 初始化DeepSearch代理
    def __init__(
        self,
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        max_iter: int = 3,
        route_collection: bool = True,
        text_window_splitter: bool = True,
        **kwargs,
    ):
        """
        Initialize the DeepSearch agent.

        Args:
            llm: The language model to use for generating answers.
            embedding_model: The embedding model to use for query embedding.
            vector_db: The vector database to search for relevant documents.
            max_iter: The maximum number of iterations for the search process.
            route_collection: Whether to use a collection router for search.
            text_window_splitter: Whether to use text_window splitter.
            **kwargs: Additional keyword arguments for customization.
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.max_iter = max_iter
        self.route_collection = route_collection
        self.collection_router = CollectionRouter(
            llm=self.llm, vector_db=self.vector_db, dim=embedding_model.dimension
        )
        self.text_window_splitter = text_window_splitter

    # 生成子查询
    def _generate_sub_queries(self, original_query: str) -> Tuple[List[str], int]:
        chat_response = self.llm.chat(
            messages=[
                {"role": "user", "content": SUB_QUERY_PROMPT.format(original_query=original_query)}
            ]
        )
        response_content = self.llm.remove_think(chat_response.content)
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    # 从向量数据库搜索文档块
    async def _search_chunks_from_vectordb(self, query: str, sub_queries: List[str]):
        consume_tokens = 0
        if self.route_collection:
            selected_collections, n_token_route = self.collection_router.invoke(
                query=query, dim=self.embedding_model.dimension
            )
        else:
            selected_collections = self.collection_router.all_collections
            n_token_route = 0
        consume_tokens += n_token_route

        all_retrieved_results = []
        query_vector = self.embedding_model.embed_query(query)
        for collection in selected_collections:
            log.color_print(f"<search> Search [{query}] in [{collection}]...  </search>\n")
            retrieved_results = self.vector_db.search_data(
                collection=collection, vector=query_vector, query_text=query
            )
            if not retrieved_results or len(retrieved_results) == 0:
                log.color_print(
                    f"<search> No relevant document chunks found in '{collection}'! </search>\n"
                )
                continue
            accepted_chunk_num = 0
            references = set()
            for retrieved_result in retrieved_results:
                chat_response = self.llm.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": RERANK_PROMPT.format(
                                query=[query] + sub_queries,
                                retrieved_chunk=f"<chunk>{retrieved_result.text}</chunk>",
                            ),
                        }
                    ]
                )
                consume_tokens += chat_response.total_tokens
                response_content = self.llm.remove_think(chat_response.content).strip()
                if "YES" in response_content and "NO" not in response_content:
                    all_retrieved_results.append(retrieved_result)
                    accepted_chunk_num += 1
                    references.add(retrieved_result.reference)
            if accepted_chunk_num > 0:
                log.color_print(
                    f"<search> Accept {accepted_chunk_num} document chunk(s) from references: {list(references)} </search>\n"
                )
            else:
                log.color_print(
                    f"<search> No document chunk accepted from '{collection}'! </search>\n"
                )
        return all_retrieved_results, consume_tokens

    # 生成缺失查询
    def _generate_gap_queries(
        self, original_query: str, all_sub_queries: List[str], all_chunks: List[RetrievalResult]
    ) -> Tuple[List[str], int]:
        reflect_prompt = REFLECT_PROMPT.format(
            question=original_query,
            mini_questions=all_sub_queries,
            mini_chunk_str=self._format_chunk_texts([chunk.text for chunk in all_chunks])
            if len(all_chunks) > 0
            else "NO RELATED CHUNKS FOUND.",
        )
        chat_response = self.llm.chat([{"role": "user", "content": reflect_prompt}])
        response_content = self.llm.remove_think(chat_response.content)
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    # 从知识库检索相关文档
    def retrieve(self, original_query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """
        Retrieve relevant documents from the knowledge base for the given query.

        This method performs a deep search through the vector database to find
        the most relevant documents for answering the query.

        Args:
            original_query (str): The query to search for.
            **kwargs: Additional keyword arguments for customizing the retrieval.

        Returns:
            Tuple[List[RetrievalResult], int, dict]: A tuple containing:
                - A list of retrieved document results
                - The token usage for the retrieval operation
                - Additional information about the retrieval process
        """
        return asyncio.run(self.async_retrieve(original_query, **kwargs))

    # 异步检索相关文档
    async def async_retrieve(
        self, original_query: str, **kwargs
    ) -> Tuple[List[RetrievalResult], int, dict]:
        # 获取最大迭代次数，如果未指定则使用默认值
        max_iter = kwargs.pop("max_iter", self.max_iter)
        ### 子查询生成阶段 ###
        # 记录原始查询到日志
        log.color_print(f"<query> {original_query} </query>\n")
        # 初始化所有搜索结果的累积列表
        all_search_res = []
        # 初始化所有子查询的累积列表
        all_sub_queries = []
        # 初始化总token消耗计数器
        total_tokens = 0

        # 第一步：调用LLM将原始查询分解为多个子查询
        sub_queries, used_token = self._generate_sub_queries(original_query)
        # 累加子查询生成的token消耗
        total_tokens += used_token
        # 检查是否成功生成了子查询
        if not sub_queries:
            # 如果没有生成子查询，记录警告并退出
            log.color_print("No sub queries were generated by the LLM. Exiting.")
            return [], total_tokens, {}
        else:
            # 记录分解结果到日志
            log.color_print(
                f"<think> Break down the original query into new sub queries: {sub_queries}</think>\n"
            )
        # 将生成的子查询添加到累积列表中
        all_sub_queries.extend(sub_queries)
        # 初始化当前迭代要处理的子查询列表
        sub_gap_queries = sub_queries

        for iter in range(max_iter):
            log.color_print(f">> Iteration: {iter + 1}\n")
            search_res_from_vectordb = []
            search_res_from_internet = []  # TODO

            # 创建所有搜索任务
            # Create all search tasks
            search_tasks = [
                self._search_chunks_from_vectordb(query, sub_gap_queries)
                for query in sub_gap_queries
            ]
            # 并行执行所有任务并等待结果
            # Execute all tasks in parallel and wait for results
            search_results = await asyncio.gather(*search_tasks)
            # 合并所有结果
            # Merge all results
            for result in search_results:
                search_res, consumed_token = result
                total_tokens += consumed_token
                search_res_from_vectordb.extend(search_res)

            search_res_from_vectordb = deduplicate_results(search_res_from_vectordb)
            # search_res_from_internet = deduplicate_results(search_res_from_internet)
            all_search_res.extend(search_res_from_vectordb + search_res_from_internet)
            if iter == max_iter - 1:
                log.color_print("<think> Exceeded maximum iterations. Exiting. </think>\n")
                break
            ### REFLECTION & GET GAP QUERIES ###
            log.color_print("<think> Reflecting on the search results... </think>\n")
            sub_gap_queries, consumed_token = self._generate_gap_queries(
                original_query, all_sub_queries, all_search_res
            )
            total_tokens += consumed_token
            if not sub_gap_queries or len(sub_gap_queries) == 0:
                log.color_print("<think> No new search queries were generated. Exiting. </think>\n")
                break
            else:
                log.color_print(
                    f"<think> New search queries for next iteration: {sub_gap_queries} </think>\n"
                )
                all_sub_queries.extend(sub_gap_queries)

        all_search_res = deduplicate_results(all_search_res)
        additional_info = {"all_sub_queries": all_sub_queries}
        return all_search_res, total_tokens, additional_info

    # 查询代理并基于检索文档生成答案
    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        Query the agent and generate an answer based on retrieved documents.

        This method retrieves relevant documents and uses the language model
        to generate a comprehensive answer to the query.

        Args:
            query (str): The query to answer.
            **kwargs: Additional keyword arguments for customizing the query process.

        Returns:
            Tuple[str, List[RetrievalResult], int]: A tuple containing:
                - The generated answer
                - A list of retrieved document results
                - The total token usage
        """
        all_retrieved_results, n_token_retrieval, additional_info = self.retrieve(query, **kwargs)
        if not all_retrieved_results or len(all_retrieved_results) == 0:
            return f"No relevant information found for query '{query}'.", [], n_token_retrieval
        all_sub_queries = additional_info["all_sub_queries"]
        chunk_texts = []
        for chunk in all_retrieved_results:
            if self.text_window_splitter and "wider_text" in chunk.metadata:
                chunk_texts.append(chunk.metadata["wider_text"])
            else:
                chunk_texts.append(chunk.text)
        log.color_print(
            f"<think> Summarize answer from all {len(all_retrieved_results)} retrieved chunks... </think>\n"
        )
        summary_prompt = SUMMARY_PROMPT.format(
            question=query,
            mini_questions=all_sub_queries,
            mini_chunk_str=self._format_chunk_texts(chunk_texts),
        )
        chat_response = self.llm.chat([{"role": "user", "content": summary_prompt}])
        log.color_print("\n==== FINAL ANSWER====\n")
        log.color_print(self.llm.remove_think(chat_response.content))
        return (
            self.llm.remove_think(chat_response.content),
            all_retrieved_results,
            n_token_retrieval + chat_response.total_tokens,
        )

    # 格式化块文本
    def _format_chunk_texts(self, chunk_texts: List[str]) -> str:
        chunk_str = ""
        for i, chunk in enumerate(chunk_texts):
            chunk_str += f"""<chunk_{i}>\n{chunk}\n</chunk_{i}>\n"""
        return chunk_str
