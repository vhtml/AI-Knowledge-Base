"""模型调用层：结构化决策 + 流式回复。

两种能力：

- `complete_json()` —— 让模型输出结构化 JSON，决定这一轮要执行什么动作、
  要落库什么内容。决策和执行分开：模型只负责"想"，改状态由代码来做，
  这样留痕和状态永远是真的，不会出现模型说改了但其实没改的情况。
- `stream()` —— 逐段吐出回复增量，用于 SSE 逐字推给前端。

配置从环境变量读（`.env` 放在 `Demos/` 下）：
`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`，走 OpenAI 兼容接口。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator

# 决策动作。模型只能从这几个里挑。
ACT_GOAL = "record_goal"
ACT_PLAN = "generate_plan"
ACT_DONE = "record_done"
ACT_REVIEW = "review"
ACT_QUERY = "query"
ACT_ANSWER = "answer"

ACT_ALL = (
    ACT_GOAL,
    ACT_PLAN,
    ACT_DONE,
    ACT_REVIEW,
    ACT_QUERY,
    ACT_ANSWER,
)

DECIDE_SYSTEM_PROMPT = """你是一个学习规划助手。你要读懂用户这一句话，
决定这轮该执行什么动作，并把要落库的内容一并给出。

可选动作（只选一个）：
- record_goal    用户明确说想学/要学某样东西 → 把那样东西记成目标
- generate_plan  用户要安排/排计划 → 为已有目标排出每日待办
- record_done    用户说某件事做完了
- review         用户要复盘/回顾今天
- query          用户要查进度、或问记忆里有什么
- answer         其余全部情况（闲聊、提问、寻求建议）

判断动作的要点：
- 「我想学 X」「我要学 X」「想学一下 X」是 record_goal，
  title 只填 X 本身，不要把「我想学」一起填进去
  例：用户说「我想学 Python 数据分析」→ title 填 "Python 数据分析"
- 「我每天只有 N 分钟」「我一天能学 N 小时」是**时间预算**，不是目标：
  action 填 answer，另外在 memory 里加一条 category="preference"、
  text 写清楚每天有多少分钟的条目，promote 填 true
- 「帮我排个计划」「安排一下这周」是 generate_plan
- 「今天搞定了那个环境配置」是 record_done
- 「今天怎么样」「复盘一下」是 review
- 「我进度如何」「我还记得什么」是 query

其他规则：
1. 只输出一个 JSON 对象，不要任何解释文字，不要用 ``` 包裹
2. memory 是这一轮值得记下来的内容，每条形如
   {"text": "...", "category": "preference|note|goal|weakness|strength",
    "importance": 1到5, "promote": true或false}
   promote 为 true 表示这条值得长期保留（例如时间预算、学习目标、长期有效的偏好）；
   为 false 表示只是当天的事，进短期工作区就够了
3. action 是 generate_plan 时，plan 里的每一项形如
   {"date": "YYYY-MM-DD", "title": "...", "minutes": 30, "milestone": false}
   date 必须是 YYYY-MM-DD 格式，日期从「今天」往后排
4. action 是 record_goal 时，goal 形如
   {"title": "...", "deadline": "YYYY-MM-DD 或空字符串", "tags": ["..."]}
5. action 是 record_done 时，done 形如 {"title": "要标记的那件事的标题"}
6. action 是 review 时，review 形如
   {"highlights": ["..."], "blockers": ["..."], "next_step": "..."}
7. 不要编造状态和记忆里没有的信息。用户没给的信息就留空
"""

REPLY_SYSTEM_PROMPT = """你是一个学习规划助手。你已经处理完用户这一轮的要求，
现在要把结果讲给用户听。

要求：
1. 用中文，像真人说话，不要客套不要寒暄
2. 只讲「本轮已发生」里真实发生的事，不要编造
3. 用户要的动作你这轮没做（比如挪动任务、删除内容、改预算），
   就直说这次没做、需要再试一次——**不要说成做了**。
   说成做了就是在骗用户，比没做完更糟
4. 这次排期按什么依据排的，就如实说出来（记忆里读到的时间预算 / 默认值）
5. 记忆里有相关的信息就自然地用上
6. 不要复述 JSON，不要罗列字段名，直接说人话
"""


@dataclass
class Delta:
    """流式输出的一段增量。"""

    text: str
    finish_reason: str | None = None


class LLMClient:
    """基于 OpenAI 兼容接口的客户端。"""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(
            base_url=base_url or None,
            api_key=api_key,
            timeout=timeout,
        )

    def stream(self, api_messages: list[dict[str, str]]) -> Iterator[Delta]:
        """逐段吐出文本增量。"""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
            "stream": True,
        }

        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        for chunk in self._client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            text = choice.delta.content or ""

            if text:
                yield Delta(text=text)

            if choice.finish_reason:
                yield Delta(text="", finish_reason=choice.finish_reason)

    def complete(self, api_messages: list[dict[str, str]]) -> str:
        """一次性拿到完整文本。"""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
        }

        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        response = self._client.chat.completions.create(**kwargs)

        if not response.choices:
            return ""

        return response.choices[0].message.content or ""

    def complete_json(self, api_messages: list[dict[str, str]]) -> dict[str, Any]:
        """要一份结构化决策。

        模型不一定守规矩，所以这里做三层兜底：剥代码块包裹 → 截取最外层
        大括号 → 兜底解析失败时返回空字典，由调用方按"没给动作"处理。
        """

        raw = self.complete(api_messages).strip()
        parsed = _extract_json(raw)

        return parsed if isinstance(parsed, dict) else {}


def _extract_json(raw: str) -> Any:
    """从模型输出里抠出 JSON。

    模型很爱加 ```json 包裹或者在前后说两句寒暄，这里都容忍。
    """

    if not raw:
        return None

    text = raw.strip()

    if text.startswith("```"):
        # ```json\n{...}\n``` → 只留中间
        parts = text.split("```")

        for part in parts:
            candidate = part.strip()

            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()

            if candidate.startswith("{"):
                text = candidate
                break

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def load_env_config() -> dict[str, str]:
    """从环境变量读取模型配置（`.env` 由调用方加载）。"""

    return {
        "base_url": os.getenv("LLM_BASE_URL") or "",
        "api_key": os.getenv("LLM_API_KEY") or "",
        "model": os.getenv("LLM_MODEL") or "gpt-4o-mini",
    }


def build_client_from_env() -> LLMClient | None:
    """按环境变量装配一个客户端。

    没有 API Key 就返回 None——调用方据此退回规则模式，
    服务照常可用，只是回答由模板生成。
    """

    config = load_env_config()

    if not config["api_key"]:
        return None

    return LLMClient(
        model=config["model"],
        base_url=config["base_url"] or None,
        api_key=config["api_key"],
    )
