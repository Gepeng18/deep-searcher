import os
from typing import List, Union

from tqdm import tqdm

# 从deepsearcher模块导入配置，全局变量通过configuration模块访问
from deepsearcher import configuration
from deepsearcher.loader.splitter import split_docs_to_chunks


# 从本地文件加载知识到向量数据库
def load_from_local_files(
    paths_or_directory: Union[str, List[str]],
    collection_name: str = None,
    collection_description: str = None,
    force_new_collection: bool = False,
    chunk_size: int = 1500,
    chunk_overlap: int = 100,
    batch_size: int = 256,
):
    """
    从本地文件或目录加载知识到向量数据库。

    此函数处理指定路径或目录中的文件，将它们分割成块，嵌入这些块，
    并将它们存储在向量数据库中。这是DeepSearcher系统数据摄入的主要方法之一。

    参数:
        paths_or_directory: 要加载的文件或目录的单个路径或路径列表。
        collection_name: 存储数据的集合名称。如果为None，则使用默认集合。
        collection_description: 集合的描述。如果为None，则不设置描述。
        force_new_collection: 如果为True，则删除现有集合并创建一个新的集合。
        chunk_size: 每个块的大小，以字符数计算。
        chunk_overlap: 块之间重叠的字符数。
        batch_size: 嵌入过程中一次处理的块数。

    引发:
        FileNotFoundError: 如果指定的任何路径不存在。
    """
    # 获取配置的向量数据库实例
    vector_db = configuration.vector_db
    # 如果未指定集合名称，使用默认集合名称
    if collection_name is None:
        collection_name = vector_db.default_collection
    # 将集合名称中的空格和连字符替换为下划线，确保集合名称的有效性
    collection_name = collection_name.replace(" ", "_").replace("-", "_")
    # 获取配置的嵌入模型实例
    embedding_model = configuration.embedding_model
    # 获取配置的文件加载器实例
    file_loader = configuration.file_loader
    # 初始化向量数据库集合，指定维度、集合名称、描述和是否强制新建
    vector_db.init_collection(
        dim=embedding_model.dimension,  # 嵌入向量的维度
        collection=collection_name,  # 集合名称
        description=collection_description,  # 集合描述
        force_new_collection=force_new_collection,  # 是否强制新建集合
    )
    # 确保paths_or_directory是列表格式，便于统一处理
    if isinstance(paths_or_directory, str):
        paths_or_directory = [paths_or_directory]
    # 初始化文档列表，用于收集所有加载的文档
    all_docs = []
    # 使用tqdm显示加载进度
    for path in tqdm(paths_or_directory, desc="Loading files"):
        # 检查路径是否存在，如果不存在则抛出异常
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: File or directory '{path}' does not exist.")
        # 根据路径类型选择加载方法：目录使用load_directory，文件使用load_file
        if os.path.isdir(path):
            docs = file_loader.load_directory(path)  # 加载整个目录
        else:
            docs = file_loader.load_file(path)  # 加载单个文件
        # 将加载的文档添加到总文档列表中
        all_docs.extend(docs)
    # 将文档分割成带有上下文窗口的文本块
    chunks = split_docs_to_chunks(
        all_docs,  # 所有加载的文档
        chunk_size=chunk_size,  # 块大小
        chunk_overlap=chunk_overlap,  # 块重叠大小
    )

    # 对文本块进行批量嵌入处理
    chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)
    # 将嵌入后的数据插入到向量数据库的指定集合中
    vector_db.insert_data(collection=collection_name, chunks=chunks)


# 从网站加载知识到向量数据库
def load_from_website(
    urls: Union[str, List[str]],
    collection_name: str = None,
    collection_description: str = None,
    force_new_collection: bool = False,
    chunk_size: int = 1500,
    chunk_overlap: int = 100,
    batch_size: int = 256,
    **crawl_kwargs,
):
    """
    从网站加载知识到向量数据库。

    此函数抓取指定的URL，处理内容，将其分割成块，嵌入这些块，
    并将它们存储在向量数据库中。这是DeepSearcher系统从网络获取数据的核心方法。

    参数:
        urls: 要抓取的单个URL或URL列表。
        collection_name: 存储数据的集合名称。如果为None，则使用默认集合。
        collection_description: 集合的描述。如果为None，则不设置描述。
        force_new_collection: 如果为True，则删除现有集合并创建一个新的集合。
        chunk_size: 每个块的大小，以字符数计算。
        chunk_overlap: 块之间重叠的字符数。
        batch_size: 嵌入过程中一次处理的块数。
        **crawl_kwargs: 传递给网络爬虫的其他关键字参数。
    """
    # 确保urls是列表格式，便于统一处理
    if isinstance(urls, str):
        urls = [urls]
    # 获取配置的向量数据库实例
    vector_db = configuration.vector_db
    # 获取配置的嵌入模型实例
    embedding_model = configuration.embedding_model
    # 获取配置的网络爬虫实例
    web_crawler = configuration.web_crawler

    # 初始化向量数据库集合
    vector_db.init_collection(
        dim=embedding_model.dimension,  # 嵌入向量的维度
        collection=collection_name,  # 集合名称
        description=collection_description,  # 集合描述
        force_new_collection=force_new_collection,  # 是否强制新建集合
    )

    # 使用网络爬虫抓取指定的URL列表，获取所有文档
    all_docs = web_crawler.crawl_urls(urls, **crawl_kwargs)

    # 将抓取的文档分割成带有上下文窗口的文本块
    chunks = split_docs_to_chunks(
        all_docs,  # 所有抓取的文档
        chunk_size=chunk_size,  # 块大小
        chunk_overlap=chunk_overlap,  # 块重叠大小
    )
    # 对文本块进行批量嵌入处理
    chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)
    # 将嵌入后的数据插入到向量数据库的指定集合中
    vector_db.insert_data(collection=collection_name, chunks=chunks)
