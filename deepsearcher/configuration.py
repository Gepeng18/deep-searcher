import os
from typing import Literal

import yaml

from deepsearcher.agent import ChainOfRAG, DeepSearch, NaiveRAG
from deepsearcher.agent.rag_router import RAGRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.loader.file_loader.base import BaseLoader
from deepsearcher.loader.web_crawler.base import BaseCrawler
from deepsearcher.vector_db.base import BaseVectorDB

# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 默认配置文件路径
DEFAULT_CONFIG_YAML_PATH = os.path.join(current_dir, "config.yaml")

# 定义支持的功能类型
FeatureType = Literal["llm", "embedding", "file_loader", "web_crawler", "vector_db"]


# DeepSearcher的配置类
class Configuration:
    """
    DeepSearcher的配置管理类。

    此类管理DeepSearcher系统中各种组件的配置设置，
    包括LLM提供商、嵌入模型、文件加载器、网络爬虫和向量数据库。
    它从YAML文件加载配置，并提供获取和设置提供商配置的方法。
    配置系统是DeepSearcher架构的核心，允许灵活地配置不同的AI服务和数据处理组件。
    """

    # 初始化Configuration对象
    def __init__(self, config_path: str = DEFAULT_CONFIG_YAML_PATH):
        """
        初始化Configuration对象。

        参数:
            config_path: 配置YAML文件的路径。默认为项目根目录下的config.yaml文件。
        """
        # 初始化默认配置
        # 从YAML文件加载配置数据
        config_data = self.load_config_from_yaml(config_path)
        # 设置提供商配置，包含各种AI服务和数据处理的提供商信息
        self.provide_settings = config_data["provide_settings"]
        # 设置查询配置，包含RAG查询的相关参数
        self.query_settings = config_data["query_settings"]
        # 设置加载配置，包含数据加载和处理的配置参数
        self.load_settings = config_data["load_settings"]

    # 从YAML文件加载配置
    def load_config_from_yaml(self, config_path: str):
        """
        从YAML配置文件加载配置信息。

        YAML（YAML Ain't Markup Language）是一种人类可读的数据序列化标准，
        在DeepSearcher中用于存储各种组件的配置参数，如API密钥、模型名称等。

        参数:
            config_path: 配置YAML文件的路径。

        返回:
            加载的配置数据，以字典形式返回。
        """
        # 以只读模式打开配置文件
        with open(config_path, "r") as file:
            # 使用安全的YAML加载器解析文件内容
            # yaml.safe_load() 比 yaml.load() 更安全，可以防止执行任意代码
            return yaml.safe_load(file)

    # 设置指定功能的提供商及其配置
    def set_provider_config(self, feature: FeatureType, provider: str, provider_configs: dict):
        """
        为指定功能设置提供商及其配置信息。

        此方法允许动态配置DeepSearcher的各个组件，例如：
        - llm: 设置语言模型提供商（如OpenAI、DeepSeek等）
        - embedding: 设置嵌入模型提供商
        - file_loader: 设置文件加载器类型
        - web_crawler: 设置网络爬虫类型
        - vector_db: 设置向量数据库类型

        参数:
            feature: 要配置的功能类型（如'llm'、'file_loader'、'web_crawler'）。
            provider: 提供商名称（如'openai'、'deepseek'）。
            provider_configs: 包含提供商特定配置的字典（如API密钥、模型参数等）。

        引发:
            ValueError: 如果功能类型不受支持。
        """
        # 检查功能类型是否在支持的功能列表中
        if feature not in self.provide_settings:
            raise ValueError(f"Unsupported feature: {feature}")

        # 设置指定功能的提供商名称
        self.provide_settings[feature]["provider"] = provider
        # 设置指定功能的详细配置参数
        self.provide_settings[feature]["config"] = provider_configs

    # 获取指定功能的当前提供商及其配置
    def get_provider_config(self, feature: FeatureType):
        """
        获取指定功能的当前提供商及其配置信息。

        此方法用于查询系统中各个组件的当前配置状态，
        在调试、监控或动态调整配置时非常有用。

        参数:
            feature: 要检索的功能类型（如'llm'、'file_loader'、'web_crawler'）。

        返回:
            包含提供商及其配置的字典。

        引发:
            ValueError: 如果功能类型不受支持。
        """
        # 检查功能类型是否受支持
        if feature not in self.provide_settings:
            raise ValueError(f"Unsupported feature: {feature}")

        # 返回指定功能的完整配置信息（包括提供商名称和详细配置）
        return self.provide_settings[feature]


# DeepSearcher系统中各种模块的工厂类
class ModuleFactory:
    """
    DeepSearcher系统中各种模块的工厂类。

    此类基于配置设置创建LLM、嵌入模型、文件加载器、网络爬虫
    和向量数据库的实例。工厂模式在这里的应用使得系统具有良好的可扩展性和灵活性，
    可以根据配置文件动态创建不同类型的组件实例。
    """

    # 初始化ModuleFactory
    def __init__(self, config: Configuration):
        """
        初始化ModuleFactory。

        参数:
            config: 包含提供商设置的Configuration对象。
        """
        # 保存配置对象引用，用于后续创建模块实例时使用
        self.config = config

    # 基于功能和模块名称创建模块实例
    def _create_module_instance(self, feature: FeatureType, module_name: str):
        """
        基于功能类型和模块名称创建模块实例。

        此方法使用Python的动态导入机制，根据配置文件中指定的提供商名称
        动态加载相应的类并创建实例。这是工厂模式的核心实现。

        参数:
            feature: 功能类型（如'llm'、'embedding'）。
            module_name: 要从中导入的模块名称。

        返回:
            指定模块的实例。
        """
        # 示例说明：
        # feature = "file_loader"
        # module_name = "deepsearcher.loader.file_loader"
        # 从配置中获取指定功能的提供商类名（如"PDFLoader"）
        class_name = self.config.provide_settings[feature]["provider"]
        # 动态导入模块，使用__import__函数和fromlist参数
        module = __import__(module_name, fromlist=[class_name])
        # 从导入的模块中获取类对象
        class_ = getattr(module, class_name)
        # 使用配置中的参数创建类的实例并返回
        return class_(**self.config.provide_settings[feature]["config"])

    # 创建语言模型实例
    def create_llm(self) -> BaseLLM:
        """
        创建语言模型实例。

        根据配置文件中指定的LLM提供商（如OpenAI、DeepSeek等）创建相应的语言模型实例。
        这是RAG系统中最重要的组件之一，负责理解用户查询和生成答案。

        返回:
            BaseLLM实现的实例。
        """
        # 使用内部方法创建LLM实例，从deepsearcher.llm模块中导入
        return self._create_module_instance("llm", "deepsearcher.llm")

    # 创建嵌入模型实例
    def create_embedding(self) -> BaseEmbedding:
        """
        创建嵌入模型实例。

        根据配置文件中指定的嵌入模型提供商创建相应的嵌入模型实例。
        嵌入模型用于将文本转换为向量表示，是向量检索的基础。

        返回:
            BaseEmbedding实现的实例。
        """
        # 使用内部方法创建嵌入模型实例，从deepsearcher.embedding模块中导入
        return self._create_module_instance("embedding", "deepsearcher.embedding")

    # 创建文件加载器实例
    def create_file_loader(self) -> BaseLoader:
        """
        创建文件加载器实例。

        根据配置文件中指定的文件加载器类型创建相应的文件加载器实例。
        文件加载器负责将各种格式的文件（如PDF、TXT、MD等）转换为可处理的文档对象。

        返回:
            BaseLoader实现的实例。
        """
        # 使用内部方法创建文件加载器实例，从deepsearcher.loader.file_loader模块中导入
        return self._create_module_instance("file_loader", "deepsearcher.loader.file_loader")

    # 创建网络爬虫实例
    def create_web_crawler(self) -> BaseCrawler:
        """
        创建网络爬虫实例。

        根据配置文件中指定的网络爬虫类型创建相应的网络爬虫实例。
        网络爬虫负责从网页抓取内容并转换为可处理的文档对象。

        返回:
            BaseCrawler实现的实例。
        """
        # 使用内部方法创建网络爬虫实例，从deepsearcher.loader.web_crawler模块中导入
        return self._create_module_instance("web_crawler", "deepsearcher.loader.web_crawler")

    # 创建向量数据库实例
    def create_vector_db(self) -> BaseVectorDB:
        """
        创建向量数据库实例。

        根据配置文件中指定的向量数据库类型创建相应的向量数据库实例。
        向量数据库用于存储和检索文本的向量表示，是RAG系统的核心存储组件。

        返回:
            BaseVectorDB实现的实例。
        """
        # 使用内部方法创建向量数据库实例，从deepsearcher.vector_db模块中导入
        return self._create_module_instance("vector_db", "deepsearcher.vector_db")


# 默认配置实例
config = Configuration()

# 全局模块实例变量，初始为None，在init_config函数中被初始化
module_factory: ModuleFactory = None  # 模块工厂实例
llm: BaseLLM = None  # 语言模型实例
embedding_model: BaseEmbedding = None  # 嵌入模型实例
file_loader: BaseLoader = None  # 文件加载器实例
vector_db: BaseVectorDB = None  # 向量数据库实例
web_crawler: BaseCrawler = None  # 网络爬虫实例
default_searcher: RAGRouter = None  # 默认RAG路由器实例
naive_rag: NaiveRAG = None  # 朴素RAG实例


# 初始化全局配置和创建所有必需的模块实例
def init_config(config: Configuration):
    """
    初始化全局配置并创建所有必需的模块实例。

    此函数初始化LLM、嵌入模型、文件加载器、网络爬虫、向量数据库和RAG代理的全局变量。
    这是一个系统初始化的关键函数，确保所有组件都被正确实例化和配置。

    参数:
        config: 用于初始化的Configuration对象。
    """
    # 声明要修改的全局变量
    global \
        module_factory, \
        llm, \
        embedding_model, \
        file_loader, \
        vector_db, \
        web_crawler, \
        default_searcher, \
        naive_rag

    # 创建模块工厂实例
    module_factory = ModuleFactory(config)

    # 通过工厂方法创建各个组件实例
    llm = module_factory.create_llm()  # 创建语言模型
    embedding_model = module_factory.create_embedding()  # 创建嵌入模型
    file_loader = module_factory.create_file_loader()  # 创建文件加载器
    web_crawler = module_factory.create_web_crawler()  # 创建网络爬虫
    vector_db = module_factory.create_vector_db()  # 创建向量数据库

    # 创建默认的RAG路由器，包含两种不同的RAG策略
    default_searcher = RAGRouter(
        llm=llm,
        rag_agents=[
            # 深度搜索代理 - 适合复杂查询和报告生成
            DeepSearch(
                llm=llm,
                embedding_model=embedding_model,
                vector_db=vector_db,
                max_iter=config.query_settings["max_iter"],  # 最大迭代次数
                route_collection=True,  # 启用集合路由
                text_window_splitter=True,  # 启用文本窗口分割
            ),
            # 链式RAG代理 - 适合处理具体的事实查询和多跳问题
            ChainOfRAG(
                llm=llm,
                embedding_model=embedding_model,
                vector_db=vector_db,
                max_iter=config.query_settings["max_iter"],  # 最大迭代次数
                route_collection=True,  # 启用集合路由
                text_window_splitter=True,  # 启用文本窗口分割
            ),
        ],
    )

    # 创建朴素RAG代理，作为简单的基准实现
    naive_rag = NaiveRAG(
        llm=llm,
        embedding_model=embedding_model,
        vector_db=vector_db,
        top_k=10,  # 检索前10个最相关的结果
        route_collection=True,  # 启用集合路由
        text_window_splitter=True,  # 启用文本窗口分割
    )
