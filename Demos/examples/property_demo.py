"""演示 @property：为什么 response.output_text 能"直接"拿到文本。

不依赖 openai SDK，用纯 Python 复刻 openai Response.output_text 的实现思路。
"""

'''
说明：对于 Node.js SDK 的常规 Responses API 响应，output_text 是 SDK 后续添加到响应对象上的普通属性，不是每次访问时才执行遍历的 getter。
python sdk的实现也同理， 事实上 type(r.output_text)值为<class 'str'>


'''

from __future__ import annotations


# ---------- 1. 原始数据结构：和服务端返回的 JSON 完全一致 ----------

OUTPUT = [
    # 第一个 item 是 reasoning，没有 content 字段
    {"type": "reasoning", "id": "rs_1", "summary": []},
    {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": "你好，", "annotations": []},
            {"type": "output_text", "text": "我是助手。", "annotations": []},
        ],
    },
]


# ---------- 2. 用 @property 把上面的 dict 包成对象 ----------

class Response:
    """模拟 openai.types.responses.Response。

    服务端只返回 output 这个列表，output_text 是 SDK 本地算出来的。
    """

    def __init__(self, data: dict):
        self._data = data          # 原始 JSON
        self.output = data["output"]  # 直接赋值给同名属性（不是 property）

    # 只读的计算属性：遍历 output，收集所有 output_text part 的 text
    @property
    def output_text(self) -> str:
        parts: list[str] = []
        for item in self.output:            # 1. 遍历每个 item
            if item.get("type") != "message":  # 2. 跳过 reasoning / function_call
                continue
            for part in item.get("content", []):  # 3. 遍历每个 content part
                if part.get("type") == "output_text":  # 4. 只要文本 part
                    parts.append(part["text"])
        return "".join(parts)

    # 演示：@property 也能模拟 setter，做"写入即校验"
    @output_text.setter
    def output_text(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("output_text 必须是 str")
        self.output[1]["content"] = [{"type": "output_text", "text": value, "annotations": []}]


# ---------- 3. 手动调用的写法（等同 response.output[1].content[0].text） ----------

def manual_extract(response: Response) -> str:
    return response.output[1]["content"][0]["text"]


if __name__ == "__main__":
    response = Response({"output": OUTPUT})

    print("output[0].content[0].text 的手写等价物：", manual_extract(response))
    print("response.output_text            ：", response.output_text)

    # property 可以像普通属性一样用，但背后是执行了一段函数
    print("类型：", type(Response.output_text))    # <class 'property'>
    print("定义：", Response.output_text.fget)     # <function Response.output_text at 0x...>

    # 调用"赋值"，会走 setter
    response.output_text = "被 setter 改写了"
    print("setter 之后：", response.output_text)

    # 没有 setter 时赋值会报 AttributeError
    class ReadOnly:
        @property
        def x(self):
            return 1

    try:
        ReadOnly().x = 2
    except AttributeError as e:
        print("只读 property 赋值报错：", e)
