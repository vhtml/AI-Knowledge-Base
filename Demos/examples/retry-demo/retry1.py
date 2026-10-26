from csv import Error
import time
import random

class ExponentialBackoffRetry:
    """指数退避重试"""

    def __init__(self, max_retries=3, base_delay=1, max_delay=30):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def retry(self, func, *args, **kwargs):
        """执行带重试的函数调用"""
        last_error = None

        for attempt in range(self.max_retries):
            print(f"第{attempt + 1}次尝试")
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_error = e

                # 检查是否可重试
                if not self.is_retryable(e):
                    raise

                # 最后一次尝试，不再重试
                if attempt == self.max_retries - 1:
                    break

                # 计算等待时间（指数退避 + 随机抖动）
                delay = min(
                    self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                    self.max_delay
                )

                print(f"第{attempt + 1}次尝试失败，{delay:.1f}秒后重试...")
                time.sleep(delay)

        # 所有重试都失败
        raise last_error

    def is_retryable(self, error):
        """判断错误是否可重试"""
        error_str = str(error).lower()
        retryable_errors = ["timeout", "connection", "busy", "temporarily"]

        print(f"error_str: {error_str}")

        for retryable_error in retryable_errors:
            if retryable_error in error_str:
                return True

        return False

# 使用示例
retry = ExponentialBackoffRetry(max_retries=3)

def call_weather_api(city, date):
    """调用天气API（模拟可能失败）"""
    # 模拟API调用
    if random.random() < 0.3:  # 30%概率失败
        raise ConnectionError("API connection timeout")
    return {"temperature": 25, "condition": "晴"}

try:
    result = retry.retry(call_weather_api, "北京", "今天")
    print(f"成功获取天气: {result}")
except Exception as e:
    print(f"获取天气失败: {e}")