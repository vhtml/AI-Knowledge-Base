"""skill_lab · Skill 制作七天教程的可运行实验。

这个包解决什么问题：把 AI/Skill/ 里的原理笔记变成能跑、能断言、能复现的实验。
每个 lesson 对应一天主题，代码里写死了当天的核心断言——跑挂了就是没学会。

用法：
    python -m skill_lab day1      跑某一天
    python -m skill_lab all       跑全部七天
    python -m skill_lab selftest  跑自检
"""
from __future__ import annotations

__all__ = ["labkit"]
