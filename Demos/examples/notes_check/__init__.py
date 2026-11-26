"""笔记结构校验工具的包入口。

这个包解决什么问题：`AI/AI-Product/` 是手工维护的长文档，
链接失效、表格列数错位、代码围栏没闭合、占位符忘记删——
这四类问题肉眼看一遍很难发现，但会让读者在某个链接上白跑一趟。

用法：
    cd Demos/examples
    uv run --project .. python -m notes_check
"""
from __future__ import annotations

from .notes_check import (
    CHAPTER_FILES,
    APPENDIX_FILES,
    INDEX_FILE,
    Report,
    assert_all_passed,
    check_all,
    notes_dir,
    repo_root,
    run,
)

__all__ = [
    "APPENDIX_FILES",
    "CHAPTER_FILES",
    "INDEX_FILE",
    "Report",
    "assert_all_passed",
    "check_all",
    "notes_dir",
    "repo_root",
    "run",
]
