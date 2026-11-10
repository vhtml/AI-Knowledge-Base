"""生活助手 Agent：天气、计算、时间三个工具。

重点在两件事上：**工具选择**（谁来决定调哪个工具）和**参数校验**
（参数不合法时怎么拦下来、怎么修、怎么回问用户）。

用法：
    cd Demos/examples
    uv run --project .. python -m life_assistant --demo
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
