---
name: demo-assistant
description: 演示技能加载。当用户要求按规范回答、分点总结、或明确提到使用技能时加载本技能。
---

使用本技能时：

1. 先用一句话说明你已加载 `demo-assistant`。
2. 用简洁条目回答，避免冗长铺垫。
3. 如果结论依赖外部事实，先调用 `web_search` 再作答。
4. 如果需要查看或修改本机文件，再调用 `run_command`。
