from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from deepsearcher.loader.splitter import Chunk


# 表示从向量数据库检索的结果
class RetrievalResult:
    """
    Represents a result retrieved from the vector database.

    This class encapsulates the information about a retrieved document,
    including its embedding, text content, reference, metadata, and similarity score.

    Attributes:
        embedding: The vector embedding of the document.
        text: The text content of the document.
        reference: A reference to the source of the document.
        metadata: Additional metadata associated with the document.
        score: The similarity score of the document to the query.
    """

    # 初始化RetrievalResult对象
    def __init__(
        self,
        embedding: np.array,
        text: str,
        reference: str,
        metadata: dict,
        score: float = 0.0,
    ):
        """
        Initialize a RetrievalResult object.

        Args:
            embedding: The vector embedding of the document.
            text: The text content of the document.
            reference: A reference to the source of the document.
            metadata: Additional metadata associated with the document.
            score: The similarity score of the document to the query. Defaults to 0.0.
        """
        self.embedding = embedding
        self.text = text
        self.reference = reference
        self.metadata = metadata
        self.score: float = score

    # 返回RetrievalResult的字符串表示
    def __repr__(self):
        """
        Return a string representation of the RetrievalResult.

        Returns:
            A string representation of the RetrievalResult object.
        """
        return f"RetrievalResult(score={self.score}, embedding={self.embedding}, text={self.text}, reference={self.reference}), metadata={self.metadata}"


# 基于文本内容移除重复的结果
def deduplicate_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """
    Remove duplicate results based on text content.

    This function removes duplicate results from a list of RetrievalResult objects
    by keeping only the first occurrence of each unique text content.

    Args:
        results: A list of RetrievalResult objects to deduplicate.

    Returns:
        A list of deduplicated RetrievalResult objects.
    """
    all_text_set = set()
    deduplicated_results = []
    for result in results:
        if result.text not in all_text_set:
            all_text_set.add(result.text)
            deduplicated_results.append(result)
    return deduplicated_results


# 表示向量数据库中集合的信息
class CollectionInfo:
    """
    Represents information about a collection in the vector database.

    This class encapsulates the name and description of a collection.

    Attributes:
        collection_name: The name of the collection.
        description: The description of the collection.
    """

    # 初始化CollectionInfo对象
    def __init__(self, collection_name: str, description: str):
        """
        Initialize a CollectionInfo object.

        Args:
            collection_name: The name of the collection.
            description: The description of the collection.
        """
        self.collection_name = collection_name
        self.description = description


# 向量数据库实现的抽象基类
class BaseVectorDB(ABC):
    """
    Abstract base class for vector database implementations.

    This class defines the interface for vector database implementations,
    including methods for initializing collections, inserting data, searching,
    listing collections, and clearing the database.

    Attributes:
        default_collection: The name of the default collection.
    """

    # 初始化BaseVectorDB对象
    def __init__(
        self,
        default_collection: str = "deepsearcher",
        *args,
        **kwargs,
    ):
        """
        Initialize a BaseVectorDB object.

        Args:
            default_collection: The name of the default collection. Defaults to "deepsearcher".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.default_collection = default_collection

    # 初始化向量数据库中的集合
    @abstractmethod
    def init_collection(
        self,
        dim: int,
        collection: str,
        description: str,
        force_new_collection=False,
        *args,
        **kwargs,
    ):
        """
        Initialize a collection in the vector database.

        Args:
            dim: The dimensionality of the vectors in the collection.
            collection: The name of the collection.
            description: The description of the collection.
            force_new_collection: If True, drop the existing collection and create a new one.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    # 向集合中插入数据
    @abstractmethod
    def insert_data(self, collection: str, chunks: List[Chunk], *args, **kwargs):
        """
        Insert data into a collection in the vector database.

        Args:
            collection: The name of the collection.
            chunks: A list of Chunk objects to insert.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    # 在集合中搜索相似向量
    @abstractmethod
    def search_data(
        self, collection: str, vector: Union[np.array, List[float]], *args, **kwargs
    ) -> List[RetrievalResult]:
        """
        Search for similar vectors in a collection.

        Args:
            collection: The name of the collection.
            vector: The query vector to search for.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A list of RetrievalResult objects representing the search results.
        """
        pass

    # 列出向量数据库中的所有集合
    def list_collections(self, *args, **kwargs) -> List[CollectionInfo]:
        """
        List all collections in the vector database.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A list of CollectionInfo objects representing the collections.
        """
        pass

    # 清空向量数据库
    @abstractmethod
    def clear_db(self, *args, **kwargs):
        """
        Clear the vector database.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass
