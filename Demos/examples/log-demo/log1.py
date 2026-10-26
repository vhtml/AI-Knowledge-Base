from loguru import logger

# 设置不同级别的日志输出文件
logger.add("logs/debug.log", level="DEBUG", rotation="10 MB", filter=lambda record: record["level"].name == "DEBUG")
logger.add("logs/info.log", level="INFO", rotation="10 MB", filter=lambda record: record["level"].name == "INFO")
logger.add("logs/warning.log", level="WARNING", rotation="10 MB", filter=lambda record: record["level"].name == "WARNING")
logger.add("logs/error.log", level="ERROR", rotation="10 MB", filter=lambda record: record["level"].name == "ERROR")
logger.add("logs/critical.log", level="CRITICAL", rotation="10 MB", filter=lambda record: record["level"].name == "CRITICAL")

# 输出不同级别的日志消息
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")