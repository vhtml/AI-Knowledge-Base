import time
import random

class CircuitBreaker:
    """熔断器"""

    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def execute(self, func, *args, **kwargs):
        """通过熔断器执行函数"""
        if self.state == "open":
            # 检查是否应该尝试恢复
            if self.should_try_reset():
                self.state = "half-open"
            else:
                raise Exception("熔断器开启，服务暂时不可用")

        try:
            result = func(*args, **kwargs)

            # 成功调用，重置状态
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0

            return result

        except Exception as e:
            self.record_failure()
            raise e

    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def should_try_reset(self):
        """检查是否应该尝试重置"""
        if self.last_failure_time is None:
            return True

        elapsed = time.time() - self.last_failure_time
        return elapsed > self.reset_timeout

# 使用示例
breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30)

def unreliable_service():
    """模拟不可靠的服务"""
    if random.random() < 0.5:
        raise Exception("服务异常")
    return "服务正常"

for i in range(10):
    try:
        result = breaker.execute(unreliable_service)
        print(f"调用{i+1}: {result}")
    except Exception as e:
        print(f"调用{i+1}: {e}")
        time.sleep(1)