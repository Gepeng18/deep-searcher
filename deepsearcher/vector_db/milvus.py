from typing import List, Optional, Union

import numpy as np
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

from deepsearcher.loader.splitter import Chunk
from deepsearcher.utils import log
from deepsearcher.vector_db.base import BaseVectorDB, CollectionInfo, RetrievalResult


# Milvus向量数据库实现类
class Milvus(BaseVectorDB):
    """Milvus类是向量数据库的子类实现。"""

    # Milvus客户端实例
    client: MilvusClient = None

    # 初始化Milvus客户端
    def __init__(
        self,
        default_collection: str = "deepsearcher",
        uri: str = "http://localhost:19530",
        token: str = "root:Milvus",
        user: str = "",
        password: str = "",
        db: str = "default",
        hybrid: bool = False,
        **kwargs,
    ):
        """
        初始化Milvus客户端。

        参数:
            default_collection (str, 可选): 默认集合名称。默认为"deepsearcher"。
            uri (str, 可选): 连接到Milvus服务器的URI。默认为"http://localhost:19530"。
            token (str, 可选): Milvus的认证令牌。默认为"root:Milvus"。
            user (str, 可选): 用于认证的用户名。默认为""。
            password (str, 可选): 用于认证的密码。默认为""。
            db (str, 可选): 数据库名称。默认为"default"。
            hybrid (bool, 可选): 是否启用混合搜索。默认为False。
            **kwargs: 传递给MilvusClient的其他关键字参数。
        """
        # 调用父类构造函数
        super().__init__(default_collection)
        # 设置默认集合名称
        self.default_collection = default_collection
        # 创建Milvus客户端连接，设置30秒超时
        self.client = MilvusClient(
            uri=uri, user=user, password=password, token=token, db_name=db, timeout=30, **kwargs
        )

        # 设置是否启用混合搜索
        self.hybrid = hybrid

    # 初始化Milvus中的集合
    def init_collection(
        self,
        dim: int,
        collection: Optional[str] = "deepsearcher",
        description: Optional[str] = "",
        force_new_collection: bool = False,
        text_max_length: int = 65_535,
        reference_max_length: int = 2048,
        metric_type: str = "L2",
        *args,
        **kwargs,
    ):
        """
        Initialize a collection in Milvus.

        Args:
            dim (int): Dimension of the vector embeddings.
            collection (Optional[str], optional): Collection name. Defaults to "deepsearcher".
            description (Optional[str], optional): Collection description. Defaults to "".
            force_new_collection (bool, optional): Whether to force create a new collection if it already exists. Defaults to False.
            text_max_length (int, optional): Maximum length for text field. Defaults to 65_535.
            reference_max_length (int, optional): Maximum length for reference field. Defaults to 2048.
            metric_type (str, optional): Metric type for vector similarity search. Defaults to "L2".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        # 检查集合名称是否为空，如果为空则使用默认集合名称
        if not collection:
            collection = self.default_collection
        # 检查描述是否为None，如果是则设置为空字符串
        if description is None:
            description = ""

        # 保存度量类型用于后续的相似度搜索
        self.metric_type = metric_type

        # 尝试创建集合，包含错误处理逻辑
        try:
            # 检查集合是否已存在，设置5秒超时
            has_collection = self.client.has_collection(collection, timeout=5)
            # 如果强制新建集合且集合已存在，则先删除现有集合
            if force_new_collection and has_collection:
                self.client.drop_collection(collection)
            # 如果集合已存在且不强制新建，则直接返回
            elif has_collection:
                return
            # 创建集合模式定义，禁用动态字段，启用自动ID生成
            schema = self.client.create_schema(
                enable_dynamic_field=False, auto_id=True, description=description
            )
            # 添加主键字段，使用64位整数类型
            schema.add_field("id", DataType.INT64, is_primary=True)
            # 添加向量嵌入字段，指定向量维度
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)

            # 如果启用混合搜索，则添加文本字段用于全文检索
            if self.hybrid:
                # 配置文本分析器参数：使用标准分词器和转小写过滤器
                analyzer_params = {"tokenizer": "standard", "filter": ["lowercase"]}
                # 添加文本字段，支持全文搜索和分析
                schema.add_field(
                    "text",  # 字段名称
                    DataType.VARCHAR,  # 字段类型：变长字符串
                    max_length=text_max_length,  # 最大长度
                    analyzer_params=analyzer_params,  # 分析器参数
                    enable_match=True,  # 启用匹配功能
                    enable_analyzer=True,  # 启用文本分析器
                )
            else:
                schema.add_field("text", DataType.VARCHAR, max_length=text_max_length)

            schema.add_field("reference", DataType.VARCHAR, max_length=reference_max_length)
            schema.add_field("metadata", DataType.JSON)

            if self.hybrid:
                schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
                bm25_function = Function(
                    name="bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names="sparse_vector",
                )
                schema.add_function(bm25_function)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="embedding", metric_type=metric_type)

            if self.hybrid:
                index_params.add_index(
                    field_name="sparse_vector",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="BM25",
                )

            self.client.create_collection(
                collection,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
            )
            log.color_print(f"create collection [{collection}] successfully")
        except Exception as e:
            log.critical(f"fail to init db for milvus, error info: {e}")

    def insert_data(
        self,
        collection: Optional[str],
        chunks: List[Chunk],
        batch_size: int = 256,
        *args,
        **kwargs,
    ):
        """
        Insert data into a Milvus collection.

        Args:
            collection (Optional[str]): Collection name. If None, uses default_collection.
            chunks (List[Chunk]): List of Chunk objects to insert.
            batch_size (int, optional): Number of chunks to insert in each batch. Defaults to 256.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        if not collection:
            collection = self.default_collection
        texts = [chunk.text for chunk in chunks]
        references = [chunk.reference for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]

        datas = [
            {
                "embedding": embedding,
                "text": text,
                "reference": reference,
                "metadata": metadata,
            }
            for embedding, text, reference, metadata in zip(
                embeddings, texts, references, metadatas
            )
        ]
        batch_datas = [datas[i : i + batch_size] for i in range(0, len(datas), batch_size)]
        try:
            for batch_data in batch_datas:
                self.client.insert(collection_name=collection, data=batch_data)
        except Exception as e:
            log.critical(f"fail to insert data, error info: {e}")

    def search_data(
        self,
        collection: Optional[str],
        vector: Union[np.array, List[float]],
        top_k: int = 5,
        query_text: Optional[str] = None,
        *args,
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        Search for similar vectors in a Milvus collection.

        Args:
            collection (Optional[str]): Collection name. If None, uses default_collection.
            vector (Union[np.array, List[float]]): Query vector for similarity search.
            top_k (int, optional): Number of results to return. Defaults to 5.
            query_text (Optional[str], optional): Original query text for hybrid search. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[RetrievalResult]: List of retrieval results containing similar vectors.
        """
        if not collection:
            collection = self.default_collection
        try:
            use_hybrid = self.hybrid and query_text

            if use_hybrid:
                sparse_search_params = {"metric_type": "BM25"}
                sparse_request = AnnSearchRequest(
                    [query_text], "sparse_vector", sparse_search_params, limit=top_k
                )

                dense_search_params = {"metric_type": self.metric_type}
                dense_request = AnnSearchRequest(
                    [vector], "embedding", dense_search_params, limit=top_k
                )

                search_results = self.client.hybrid_search(
                    collection_name=collection,
                    reqs=[sparse_request, dense_request],
                    ranker=RRFRanker(),
                    limit=top_k,
                    output_fields=["embedding", "text", "reference", "metadata"],
                    timeout=10,
                )
            else:
                search_results = self.client.search(
                    collection_name=collection,
                    data=[vector],
                    limit=top_k,
                    output_fields=["embedding", "text", "reference", "metadata"],
                    timeout=10,
                )

            return [
                RetrievalResult(
                    embedding=b["entity"]["embedding"],
                    text=b["entity"]["text"],
                    reference=b["entity"]["reference"],
                    score=b["distance"],
                    metadata=b["entity"]["metadata"],
                )
                for a in search_results
                for b in a
            ]
        except Exception as e:
            log.critical(f"fail to search data, error info: {e}")
            return []

    def list_collections(self, *args, **kwargs) -> List[CollectionInfo]:
        """
        List all collections in the Milvus database.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[CollectionInfo]: List of collection information objects.
        """
        collection_infos = []
        dim = kwargs.pop("dim", 0)
        try:
            collections = self.client.list_collections()
            for collection in collections:
                description = self.client.describe_collection(collection)
                if dim != 0:
                    skip = False
                    for field_dict in description["fields"]:
                        if (
                            field_dict["name"] == "embedding"
                            and field_dict["type"] == DataType.FLOAT_VECTOR
                        ):
                            if field_dict["params"]["dim"] != dim:
                                skip = True
                    if skip:
                        continue
                collection_infos.append(
                    CollectionInfo(
                        collection_name=collection,
                        description=description["description"],
                    )
                )
        except Exception as e:
            log.critical(f"fail to list collections, error info: {e}")
        return collection_infos

    def clear_db(self, collection: str = "deepsearcher", *args, **kwargs):
        """
        Clear (drop) a collection from the Milvus database.

        Args:
            collection (str, optional): Collection name to drop. Defaults to "deepsearcher".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        if not collection:
            collection = self.default_collection
        try:
            self.client.drop_collection(collection)
        except Exception as e:
            log.warning(f"fail to clear db, error info: {e}")
