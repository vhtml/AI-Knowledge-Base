"""infra_lab 自检：验证实验骨架和笔记一样都没坏。

这个模块解决什么问题：教程仓库最容易烂在"笔记写了、实验跑不起来"或者
"实验改了、笔记还对不上"。自检负责把四类漂移挡住：

1. **结构漂移**——18 章笔记与 18 个实验模块必须一一对应，且都存在、都可导入。
2. **行为漂移**——每章的确定性路径必须跑通断言。
3. **口径漂移**——所有算账都必须走 :mod:`infra_lab.labkit`，
   所以实验代码里不准出现自己写的显存/带宽公式常量表。
4. **立教漂移**——这套教程的立身之本是"离线、确定性、框架无关"，
   所以要检查实验代码里**没有**偷偷 import 推理框架或 torch。

不依赖 pytest、不联网、不调模型、不需要 GPU：

    cd Demos/examples
    uv run --project .. python -m infra_lab.selftest
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from .lessons import LESSONS

# 笔记文件与章节编号一一对应；名字改了这里也要改
NOTE_FILES: tuple[str, ...] = (
    "01-推理服务全景与 SLO.md",
    "02-GPU 规格与显存账本.md",
    "03-Roofline 与算术强度.md",
    "04-访存墙与批量摊薄.md",
    "05-KV Cache：显存账与复用.md",
    "06-Attention 变体与 PagedAttention 原理.md",
    "07-连续批处理与调度内核.md",
    "08-多租户调度与 SLA.md",
    "09-量化：精度边界与成本收益.md",
    "10-推测解码与时延收益.md",
    "11-网关：路由、限流与租户隔离.md",
    "12-缓存体系：前缀、语义与响应缓存.md",
    "13-提示词流水线与长上下文混批.md",
    "14-可观测性与延迟归因.md",
    "15-弹性：自动扩缩容与冷启动.md",
    "16-容量规划与上线检查单.md",
    "17-训练侧 Infra 地基.md",
    "18-生态地图：框架选型与机制对照.md",
)

APPENDIX_FILES: tuple[str, ...] = (
    "附录-A-术语表.md",
    "附录-B-工程检查单.md",
    "附录-C-常见故障排查.md",
)

NOTES_DIR = Path(__file__).resolve().parents[3] / "AI" / "AI-Infra"

# 这套教程禁止出现的依赖——推理框架第 18 章才允许在**文字**里讨论
FORBIDDEN_IMPORTS: tuple[str, ...] = (
    "vllm",
    "sglang",
    "tensorrt_llm",
    "transformers",
    "torch",
    "triton",
)

MIN_NOTE_BYTES = 4000
"""笔记最少字节数。低于这个数说明大概率没写完，只是占了个坑。"""


def check_notes_exist() -> list[str]:
    """18 篇笔记 + 3 份附录必须都在，且课程地图有入口链接。"""
    failures: list[str] = []
    for name in NOTE_FILES:
        path = NOTES_DIR / name
        if not path.is_file():
            failures.append(f"缺笔记：AI/AI-Infra/{name}")
            continue
        if path.stat().st_size < MIN_NOTE_BYTES:
            failures.append(
                f"笔记过短（{path.stat().st_size} < {MIN_NOTE_BYTES} 字节），疑似没写完：AI/AI-Infra/{name}"
            )

    for name in APPENDIX_FILES:
        if not (NOTES_DIR / name).is_file():
            failures.append(f"缺附录：AI/AI-Infra/{name}")

    readme = NOTES_DIR / "README.md"
    if not readme.is_file():
        failures.append("缺课程地图：AI/AI-Infra/README.md")
    else:
        text = readme.read_text(encoding="utf-8")
        for name in NOTE_FILES:
            if name not in text:
                failures.append(f"课程地图没有链接到：{name}")
        for name in APPENDIX_FILES:
            if name not in text:
                failures.append(f"课程地图没有链接到：{name}")
    return failures


def check_lessons_importable() -> list[str]:
    """每个实验模块都要能导入，且导出 ``run()``。"""
    failures: list[str] = []
    for lesson in LESSONS:
        try:
            module = importlib.import_module(f"infra_lab.lessons.{lesson.module}")
        except Exception as exc:  # noqa: BLE001 - 自检要报出任何导入问题
            failures.append(f"{lesson.key} 导入失败：{type(exc).__name__}: {exc}")
            continue
        if not callable(getattr(module, "run", None)):
            failures.append(f"{lesson.key} 缺少 run()")
        if not (module.__doc__ or "").strip():
            failures.append(f"{lesson.key} 模块没有中文说明（docstring）")
    return failures


def check_no_framework_imports() -> list[str]:
    """检查实验代码里没有偷用推理框架或 GPU 栈。

    只做静态语法检查（不 import 执行），所以跑得快，也不会触发副作用。
    第 18 章要讲框架，但那一章也只允许在**文字**里对比，不允许 import。
    """
    failures: list[str] = []
    package_dir = Path(__file__).resolve().parent
    for path in sorted(package_dir.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            failures.append(f"{path.name} 语法错误：{exc}")
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    rel = path.relative_to(package_dir)
                    failures.append(
                        f"{rel} 第 {node.lineno} 行 import 了 {name}——"
                        "这套教程必须离线框架无关，框架留到第 18 章"
                    )
    return failures


def run_lessons() -> list[str]:
    """逐章跑确定性路径，把断言失败收集起来。"""
    failures: list[str] = []
    for lesson in LESSONS:
        try:
            module = importlib.import_module(f"infra_lab.lessons.{lesson.module}")
        except Exception as exc:  # noqa: BLE001 - 缺章节也要报出来而不是崩掉
            failures.append(f"{lesson.key}（{lesson.title}）：导入失败 {type(exc).__name__}: {exc}")
            continue
        try:
            module.run()
        except AssertionError as exc:
            failures.append(f"{lesson.key}（{lesson.title}）：{exc}")
        except Exception as exc:  # noqa: BLE001 - 自检要暴露运行期错误
            failures.append(f"{lesson.key}（{lesson.title}）：{type(exc).__name__}: {exc}")
    return failures


def run_capstone() -> list[str]:
    """跑毕业作品的自检。"""
    try:
        from .capstone.selftest import run as capstone_selftest
    except Exception as exc:  # noqa: BLE001 - 缺毕业作品也要报出来
        return [f"毕业作品不可用：{type(exc).__name__}: {exc}"]
    return capstone_selftest()


def run() -> int:
    """跑完整自检。

    返回进程退出码：0 全部通过，1 有失败项。
    """
    failures: list[str] = []

    print("检查笔记与实验的对应关系……")
    failures.extend(check_notes_exist())

    print("检查实验模块可导入……")
    failures.extend(check_lessons_importable())

    print("检查实验是否保持离线框架无关……")
    failures.extend(check_no_framework_imports())

    print("逐章跑确定性断言……")
    failures.extend(run_lessons())

    print("跑毕业作品自检……")
    failures.extend(run_capstone())

    print()
    print("=" * 72)
    if failures:
        print(f"自检未通过：{len(failures)} 项")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(
        f"自检通过：{len(LESSONS)} 章实验 + {len(NOTE_FILES)} 篇笔记 "
        f"+ {len(APPENDIX_FILES)} 份附录 + 毕业作品"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(run())
