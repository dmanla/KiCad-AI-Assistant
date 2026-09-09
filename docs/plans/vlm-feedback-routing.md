# VLM 视觉反馈驱动现有布线算法（最小闭环实验）

## 状态：实现完成（方案 A，issue #124）

- 方案 A 已确认并落地：`scripts/vlm_route_feedback.py` + 单元测试，未改任何现有代码。
- 验证：dry-run 冒烟 2/2 对连成（VCC 单层、GND 跨 F.Cu→B.Cu→In1.Cu 带 via）；`tests/unit/scripts/test_vlm_route_feedback.py` 18/18 通过。

## 背景与目标

验证「VLM 看图 → 语义反馈 → 现有 router 执行 → 渲染反馈」闭环的可行性，是路线 B（VLM 规划 + PNS 精确执行）的**前置最小实验**。

核心问题：VLM 对布线渲染图的反馈，能否让现有 A\* 路由器的结果更好？具体可指导维度限在当前 `RouteRequest` 暴露的旋钮：

- 布线顺序（先连哪对 pad）
- 层选择（`layer_hint`）
- 失败后换策略（`RouteFailure` 原因回喂 → 换参数重试）

**明确不做**（后续 issue）：waypoint、候选对比渲染、PNS/sidecar 引擎。

## 现状盘点（可复用资产）

| 组件 | 位置 | 状态 |
|---|---|---|
| 路由核心 | `kcaa/router/router.py::auto_route_pair` | ✅ 网格 A\* + 45° 后处理，无 shove，挡路抛 `RouteFailure` |
| 管线可视化 | `_dump_viz`（`KCAA_DUMP_ROUTE_PIPELINE=1`） | ✅ 逐阶段 JSON |
| JSON→PNG | `scripts/render_viz.py`（matplotlib+shapely） | ✅ |
| 多模态 LLM | `kicad_plugin/llm_client.py::_build_user_content` | ✅ base64 图 |
| 整板渲染 | `kcaa/tools/export_tools.py::generate_pcb_thumbnail` | ✅ kicad-cli SVG（无 pad 编号，不适用反馈循环） |
| 测试板 | `tests/integration/fixtures/test_routing_board.kicad_pcb`（3 nets）+ `.kicad_pro` | ✅ |

**缺口**（本次要补）：

1. 整板「现状图」渲染：pad 编号 + 待连 net 高亮 + 已布线段，喂给 VLM 的输入源
2. 驱动脚本：渲染 → 问 VLM → 解析语义反馈 → 映射 `RouteRequest` → 路由 → 落盘 → 再渲染循环
3. 评估指标：连成对/尝试对、每对轮次、失败后修正成功率

## 闭环设计

```
┌──────────────┐ 整板现状图(pad编号)  ┌──────────────┐
│ 渲染脚本      │ ───────────────────▶ │ VLM          │
│              │                      │ (系统提示+图) │
└──────────────┘                      └──────┬───────┘
                                              │ 语义反馈：连 X.Y-A.B
                                              │ 层 Z / 换策略
                                              ▼
                                     ┌────────────────┐
                                     │ 解析器 → RouteRequest│
                                     └──────┬─────────┘
                                            ▼
                                     auto_route_pair
                                     ├─ 成功 → pcb_route_pad_to_pad 落盘 → 渲染新状态 → 下一轮
                                     └─ 失败 → 失败 viz + RouteFailure 回喂 VLM → 重试(上限 N)
```

**VLM 输出协议（实验用）**：

```
route: <ref_a>.<pad_a> -> <ref_b>.<pad_b>
layer: <F.Cu|B.Cu|auto>
reason: 一句话理由
```

失败回喂时附加：

```
last_error: <RouteFailure 消息>
advice: <换层/换顺序/放弃该对>
```

## 候选方案

### 方案 A：独立实验脚本（推荐）

新增 `scripts/vlm_route_feedback.py`，**不改任何现有代码**。

- 整板渲染：脚本内用 sexpdata + matplotlib 画（或复用 render_viz.py 的绘图函数导出）
- 复用：`auto_route_pair`、`load_pcb/save_pcb`、`llm_client` 的 `_build_user_content`
- 测试：`tests/unit/router/` 或脚本级冒烟（渲染一张图人工确认）——实验脚本不强制单测

优点：零侵入、闭环验证快、可整体丢弃；缺点：渲染逻辑与 render_viz 有部分重复。

### 方案 B：MCP 工具集成

新增 `kcaa/tools/vlm_route_feedback_tools.py`，注册为 fastmcp 工具，走 MCP 会话。

优点：直接成为产品能力，插件界面可用；缺点：实验未验证就产品化，返工风险高；会话状态管理复杂。

### 方案 C：人工 MCP 驱动（零开发）

直接用现有工具：`generate_pcb_thumbnail` + `pcb_route_pad_to_pad` 人工循环。

优点：零开发、今天就能试；缺点：缩略图无 pad 编号、无失败回喂、无法自动化评估——只适合"感觉一下"。

**倾向：方案 A**。理由：实验目的——证明/证伪反馈闭环，不是交付产品；零侵入保证失败可整体丢弃；评估可脚本化。

## 实验设计

- **实验 1（本次）**：反馈维度限层选择 + 布线顺序 + 失败换策略。3-net 板，对比「VLM 看图决策」vs「固定顺序盲连」的连成率。
- **实验 2（后续 issue）**：若实验 1 中 VLM 频繁想表达「绕行某区域」而现有参数做不到 → 实证 waypoint 必要性，再单独设计。

## 验收标准（对应 issue #124）

1. 脚本能渲染整板现状图（pad 编号 + 待连 net + 已布线段）
2. VLM 看图输出语义反馈，解析器映射为 `RouteRequest` 参数
3. 成功 → 落盘 + 渲染新状态；失败 → 失败 viz + 原因回喂 VLM 换策略
4. 输出评估指标：连成对/尝试对、每对轮次、失败后修正成功率

## 涉及文件

- 新增：`scripts/vlm_route_feedback.py`
- 新增：本计划文档（docs/plans/vlm-feedback-routing.md）
- 不改：router / llm_client / tools 现有代码