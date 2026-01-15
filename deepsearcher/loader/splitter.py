## 句子窗口分割策略，参考：
#  https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/advanced_rag/sentence_window_with_langchain.ipynb

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 表示带有元数据和嵌入的文本块
class Chunk:
    """
    表示带有关联元数据和嵌入的文本块。

    文本块是从文档中提取的文本段，包含其引用信息、元数据以及可选的嵌入向量。
    这在检索增强生成(RAG)系统中非常重要，用于存储和管理文档片段。

    属性:
        text: 文本块的内容。
        reference: 文本块来源的引用（例如：文件路径、URL）。
        metadata: 与文本块关联的附加元数据。
        embedding: 文本块的向量嵌入（如果可用）。
    """

    # 初始化Chunk对象
    def __init__(
        self,
        text: str,
        reference: str,
        metadata: dict = None,
        embedding: List[float] = None,
    ):
        """
        初始化Chunk对象。

        参数:
            text: 文本块的内容。这是文档被分割后的核心文本。
            reference: 文本块来源的引用。可以是文件路径、URL等，用于追踪文本的原始来源。
            metadata: 与文本块关联的附加元数据。默认为空字典。可以包含如页码、位置信息等。
            embedding: 文本块的向量嵌入。默认为None。在后续处理中会被计算和填充。
        """
        # 设置文本块的核心内容
        self.text = text
        # 设置文本块的来源引用
        self.reference = reference
        # 初始化元数据，如果未提供则使用空字典
        self.metadata = metadata or {}
        # 初始化嵌入向量，如果未提供则为None
        self.embedding = embedding or None


# 从分割后的文档创建带有上下文窗口的文本块
def _sentence_window_split(
    split_docs: List[Document], original_document: Document, offset: int = 200
) -> List[Chunk]:
    """
    从分割后的文档创建带有上下文窗口的文本块。

    此函数接收已被分割成较小片段的文档，并通过在每个分割片段前后包含原始文档的文本来添加上下文，
    最多包含指定偏移量的字符。这种方法称为"句子窗口"技术，可以提供更好的检索效果。

    参数:
        split_docs: 被分割后的文档列表。每个文档包含较小的文本片段。
        original_document: 分割前的原始文档。用于提取上下文信息。
        offset: 在每个分割片段前后包含的字符数。默认为200字符。

    返回:
        带有上下文窗口的Chunk对象列表。
    """
    # 初始化空的文本块列表，用于存储结果
    chunks = []
    # 获取原始文档的完整文本内容
    original_text = original_document.page_content
    # 遍历每个分割后的文档片段
    for doc in split_docs:
        # 获取当前文档片段的文本内容
        doc_text = doc.page_content
        # 在原始文本中找到当前片段的起始位置
        start_index = original_text.index(doc_text)
        # 计算当前片段在原始文本中的结束位置
        end_index = start_index + len(doc_text) - 1
        # 提取包含上下文的更宽文本窗口
        # 从起始位置前offset个字符开始，到结束位置后offset个字符结束
        # 使用max(0, ...)确保不超出文本开头，使用min(len(original_text), ...)确保不超出文本结尾
        wider_text = original_text[
            max(0, start_index - offset) : min(len(original_text), end_index + offset)
        ]
        # 从文档元数据中提取引用信息，如果没有则使用空字符串
        reference = doc.metadata.pop("reference", "")
        # 将上下文窗口文本添加到元数据中
        doc.metadata["wider_text"] = wider_text
        # 创建Chunk对象，包含原始文本、引用和包含上下文的元数据
        chunk = Chunk(text=doc_text, reference=reference, metadata=doc.metadata)
        # 将创建的文本块添加到结果列表中
        chunks.append(chunk)
    # 返回包含上下文窗口的所有文本块
    return chunks


# 将文档分割成带有上下文窗口的文本块
def split_docs_to_chunks(
    documents: List[Document], chunk_size: int = 1500, chunk_overlap=100
) -> List[Chunk]:
    """
    将文档列表分割成带有上下文窗口的较小文本块。

    此函数将文档列表分割成较小的文本块，具有重叠的文本内容，
    并通过在每个文本块前后包含文本来为每个文本块添加上下文窗口。
    这种方法可以提高检索质量，因为每个文本块都包含了周围的上下文信息。

    参数:
        documents: 要分割的文档列表。每个文档包含完整的文本内容。
        chunk_size: 每个文本块的大小，以字符数计算。默认为1500字符。
        chunk_overlap: 文本块之间重叠的字符数。默认为100字符，用于保持上下文连续性。

    返回:
        带有上下文窗口的Chunk对象列表。
    """
    # 创建递归字符文本分割器，用于将长文本分割成较小的块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    # 初始化空列表，用于存储所有处理后的文本块
    all_chunks = []
    # 遍历每个输入文档
    for doc in documents:
        # 使用文本分割器将当前文档分割成多个较小的文档片段
        split_docs = text_splitter.split_documents([doc])
        # 为分割后的文档片段创建上下文窗口，偏移量设为300字符
        # 这将为每个片段提供前后300字符的上下文信息
        split_chunks = _sentence_window_split(split_docs, doc, offset=300)
        # 将处理后的文本块添加到总的文本块列表中
        all_chunks.extend(split_chunks)
    # 返回所有文档处理后的文本块列表
    return all_chunks
