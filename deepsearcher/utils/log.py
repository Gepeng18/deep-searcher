import logging

from termcolor import colored


# 为日志消息添加颜色的自定义格式化器
class ColoredFormatter(logging.Formatter):
    """
    A custom formatter for logging that adds colors to log messages.

    This formatter adds colors to log messages based on their level,
    making it easier to distinguish between different types of logs.

    Attributes:
        COLORS: A dictionary mapping log levels to colors.
    """

    COLORS = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "magenta",
    }

    # 使用颜色格式化日志记录
    def format(self, record):
        """
        Format a log record with colors.

        Args:
            record: The log record to format.

        Returns:
            The formatted log message with colors.
        """
        # all line in log will be colored
        log_message = super().format(record)
        return colored(log_message, self.COLORS.get(record.levelname, "white"))

        # only log level will be colored
        # levelname_colored = colored(record.levelname, self.COLORS.get(record.levelname, 'white'))
        # record.levelname = levelname_colored
        # return super().format(record)

        # only keywords will be colored
        # message = record.msg
        # for word, color in self.KEYWORDS.items():
        #     if word in message:
        #         message = message.replace(word, colored(word, color))
        # record.msg = message
        # return super().format(record)


# config log
dev_logger = logging.getLogger("dev")
dev_formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
dev_handler = logging.StreamHandler()
dev_handler.setFormatter(dev_formatter)
dev_logger.addHandler(dev_handler)
dev_logger.setLevel(logging.INFO)

progress_logger = logging.getLogger("progress")
progress_handler = logging.StreamHandler()
progress_handler.setFormatter(ColoredFormatter("%(message)s"))
progress_logger.addHandler(progress_handler)
progress_logger.setLevel(logging.INFO)

dev_mode = False


# 设置开发模式
def set_dev_mode(mode: bool):
    """
    Set the development mode.

    When in development mode, debug, info, and warning logs are displayed.
    When not in development mode, only error and critical logs are displayed.

    Args:
        mode: True to enable development mode, False to disable it.
    """
    global dev_mode
    dev_mode = mode


# 设置开发日志记录器的日志级别
def set_level(level):
    """
    Set the logging level for the development logger.

    Args:
        level: The logging level to set (e.g., logging.DEBUG, logging.INFO).
    """
    dev_logger.setLevel(level)


# 记录调试消息
def debug(message):
    """
    Log a debug message.

    Args:
        message: The message to log.
    """
    if dev_mode:
        dev_logger.debug(message)


# 记录信息消息
def info(message):
    """
    Log an info message.

    Args:
        message: The message to log.
    """
    if dev_mode:
        dev_logger.info(message)


# 记录警告消息
def warning(message):
    """
    Log a warning message.

    Args:
        message: The message to log.
    """
    if dev_mode:
        dev_logger.warning(message)


# 记录错误消息
def error(message):
    """
    Log an error message.

    Args:
        message: The message to log.
    """
    if dev_mode:
        dev_logger.error(message)


# 记录严重消息并引发RuntimeError
def critical(message):
    """
    Log a critical message and raise a RuntimeError.

    Args:
        message: The message to log.

    Raises:
        RuntimeError: Always raised with the provided message.
    """
    dev_logger.critical(message)
    raise RuntimeError(message)


# 向进度日志记录器打印彩色消息
def color_print(message, **kwargs):
    """
    Print a colored message to the progress logger.

    Args:
        message: The message to print.
        **kwargs: Additional keyword arguments to pass to the logger.
    """
    progress_logger.info(message)
