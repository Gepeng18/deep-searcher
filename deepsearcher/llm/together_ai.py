import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class TogetherAI(BaseLLM):
    """
    TogetherAI language model implementation.

    This class provides an interface to interact with various language models
    hosted on the Together AI platform.

    Website: https://www.together.ai/

    Attributes:
        model (str): The model identifier to use on Together AI platform.
        client: The Together AI client instance.
    """

    # 初始化TogetherAI语言模型客户端
    def __init__(self, model: str = "deepseek-ai/DeepSeek-R1", **kwargs):
        """
        初始化TogetherAI语言模型客户端。

        参数:
            model (str, 可选): 要使用的模型标识符。默认为"deepseek-ai/DeepSeek-R1"。
            **kwargs: 传递给Together客户端的其他关键字参数。
                - api_key: Together AI API密钥。如果未提供，则使用TOGETHER_API_KEY环境变量。
        """
        # 导入Together客户端库
        from together import Together

        # 设置要使用的模型名称
        self.model = model
        # 检查是否在参数中提供了API密钥
        if "api_key" in kwargs:
            # 如果提供了API密钥，从kwargs中提取并移除
            api_key = kwargs.pop("api_key")
        else:
            # 如果未提供，从环境变量获取
            api_key = os.getenv("TOGETHER_API_KEY")
        # 创建Together客户端实例，传入API密钥和其他配置参数
        self.client = Together(api_key=api_key, **kwargs)

    # 向TogetherAI模型发送聊天消息并获取响应
    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        向TogetherAI模型发送聊天消息并获取响应。

        参数:
            messages (List[Dict]): 消息字典列表，通常格式为
                                  [{"role": "system", "content": "..."},
                                   {"role": "user", "content": "..."}]

        返回:
            ChatResponse: 包含模型响应和token使用信息的对象。
        """
        # 调用Together AI客户端的聊天完成API
        response = self.client.chat.completions.create(
            model=self.model,  # 指定要使用的模型
            messages=messages,  # 传递消息列表
        )
        # 创建并返回ChatResponse对象，包含响应内容和token使用量
        return ChatResponse(
            content=response.choices[0].message.content,  # 提取第一个选择的消息内容
            total_tokens=response.usage.total_tokens,  # 获取总token使用量
        )
