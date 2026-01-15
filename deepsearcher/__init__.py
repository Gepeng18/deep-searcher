import os

# 忽略警告信息
# ignore the warnings
# 未找到PyTorch、TensorFlow >= 2.0或Flax。模型将不可用，仅可使用tokenizer、配置和文件/数据工具。
# None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
