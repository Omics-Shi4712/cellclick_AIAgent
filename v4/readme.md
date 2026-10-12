# CellClick V4

CellClick V4 是 CellClick 的证据优先（evidence-first）细胞类型注释协议。它把一个 cluster 的 marker 列表绑定到真实表达矩阵，在此基础上运行固定的四阶段 LLM 工作流：A1 审计 marker 证据，A2 生成候选身份，A3 通过有限的表达统计查询进行 Annotation Boost，A4 做最终的对比式裁决。

V4.4 的协议版本为 `v4.4`，prompt 版本为 `v4.4`，工作流版本为 `marker-statistics-four-agent-boost-loop-v4.4-staged-a4-evidence-decision`。A4 被拆为总体支持证据、亚型证据和最终裁决三个严格阶段；协议和 JSON contract 的 SHA-256 指纹由 `contract.py` 动态计算，可用于运行复现和缓存校验。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `graph.py` | 公共入口 `annotate_cluster`、A1/A2/A3/A4 调度、预算控制和 trace 写入 |
| `models.py` | Pydantic 输入、证据、工具请求和四阶段响应模型 |
| `expression.py` | 校验表达绑定，计算初始 marker 统计和 COSG 分数 |
| `tools.py` | 唯一允许的工具 `get_marker_statistics`，从 h5ad 查询 target-vs-rest 统计 |
| `prompts.py` | 冻结的四个 system prompt 和 canonical JSON 请求构造 |
| `ledger.py` | 将工具观察转换为可追踪的证据 ledger 条目 |
| `contract.py` | 协议 payload、protocol fingerprint 和 contract fingerprint |
| `rag.py` | 对 v3 marker reference retriever 的 V4 评分适配器；不是 `annotate_cluster` 主流程的一部分 |
| `__init__.py` | 注册 `V4_PROTOCOL` 并导出公共入口 |

## 工作流

1. **初始证据**：`collect_initial_evidence` 读取绑定的 h5ad 和 manifest。对每个提交 marker 计算 target/rest 的阳性比例、`mean_log1p_difference` 和 COSG 分数。COSG 在完整基因空间上运行，但只缓存提交 marker 的分数。
2. **A1 marker-quality auditor**：只检查提供的统计量、COSG 和固定 marker selection，不推断细胞身份。
3. **A2 identity-hypothesis agent**：最多保留 3 个候选，要求区分 broad lineage、subtype evidence 和 competing label。
4. **A3 Annotation Boost**：严格经历 `diagnosis -> query_plan -> evidence_review -> ready`。`query_plan` 只能请求能区分已命名候选的 marker 统计；查询后必须经过 `evidence_review` 才能结束。
5. **A4 final adjudicator**：只接受有可追踪支持且没有致命冲突的标签。subtype 接受还必须有 subtype evidence 和 sibling comparison。
6. **结果**：A4 `accepted` 时返回非空 `prediction`；预算耗尽或 A4 请求证据时返回 unresolved 类状态和 `prediction: null`。

COSG 与表达统计相关，协议明确禁止把 COSG 当作独立证据重复计数。参考 marker/RAG 结果也不会自动注入 A1-A4 的主流程。

## 运行前置条件

运行时需要 Python `>=3.10`、Pydantic `>=2`、OpenAI-compatible client，以及读取 h5ad 所需的 `anndata`、`numpy` 和 `cosg`。仓库包的基础依赖见上级 `pyproject.toml`；表达和 COSG 依赖应安装在实际 caller 使用的环境中。

LLM 客户端 `cellclick.llm.OpenAIChatClient` 只从环境变量读取凭据：

```bash
export OPENAI_API_KEY=...
# 可选：OpenAI-compatible relay；也兼容 OPENAI_API_BASE
export OPENAI_BASE_URL=https://example.invalid/v1
```

不要把 key、原始响应或私密配置写入 YAML、日志或版本库。

## 表达绑定 manifest

`annotate_cluster` 必须同时接收 `expression_file` 和 `expression_manifest`。manifest 至少包含以下字段：

```json
{
  "schema_version": "cellclick.v4.expression.2",
  "h5ad_path": "/absolute/path/expression.h5ad",
  "groupby": "cellclick_group_id",
  "statistics_layer": "normalization+log1p",
  "source_row_id_by_group": {"G001": "row-1"},
  "initial_markers_by_source_row_id": {"row-1": ["CD3D", "TRBC1"]}
}
```

实现会拒绝以下情况：manifest 路径与 h5ad 不一致、任务 ID 没有且只有一个分组绑定、marker 顺序/内容与快照不一致、缺少指定分组或 layer、target 组为空或覆盖全部细胞、`var_names` 重复，或 COSG 结果缺失/非法。统计必须来自 `normalization+log1p` layer；不会回退到 raw 或 marker-only 证据。

## Python API

公共入口是同步函数 `cellclick.v4.graph.annotate_cluster`：

```python
from pathlib import Path

from cellclick.llm import OpenAIChatClient
from cellclick.v4.graph import annotate_cluster

result = annotate_cluster(
    cluster_id="row-1",
    dataset="demo",
    species="human",
    tissue="Blood",
    markers=["CD3D", "TRBC1", "IL7R"],
    client=OpenAIChatClient(),
    model="gpt-4o",
    temperature=0.0,
    retries=2,
    work_dir=Path("runs/task-row-1"),
    prompt_version="v4.2",
    protocol_version="v4",
    expression_file="runs/expression.h5ad",
    expression_manifest="runs/expression.manifest.json",
)
```

`markers` 必须非空且不能有规范化后的重复值。`prompt_version` 和 `protocol_version` 不能省略或使用其他版本。函数会在返回前关闭传入的 JSON client；若使用自定义 client，应实现可选的异步 `aclose()`。

## 预算与返回值

当前固定预算如下：

| 参数 | 值 |
| --- | ---: |
| 候选数 | 3 |
| 最大 boost round | 5 |
| 每次请求重试 | 2 |
| 每轮工具请求 | 2 |
| 全流程唯一查询 gene 数 | 200 |
| 并行 cluster 数 | 1 |

返回值形如 `{"status": ..., "round_count": ..., "final_record": ...}`。`final_record` 包含 A1/A2/A3/A4 结构化结果、候选 registry、evidence ledger、工具观察、boost history 和 call counts。成功接受标签时 `status` 为 `validated`，预测位于 `final_record["prediction"]`；boost 预算耗尽时为 `unresolved` 且 prediction 为 `null`。`build_graph()` 是兼容占位函数，调用会抛出 `NotImplementedError`，V4 的公共入口只有 `annotate_cluster`。

## 工具和证据规则

工具注册表是静态且只有 `get_marker_statistics`。每个请求必须声明 task、候选 hypothesis、用途和 discriminating question；重复请求、未知 hypothesis、超过 gene 预算或跨 task 请求都会被拒绝。工具返回 `success`、`capability_unavailable`、`tool_error` 或 `protocol_error`，失败观察不能携带证据包。

所有 evidence ID、subtype evidence ID 和 sibling comparison ID 在进入下一阶段前都会验证归属。A4 的 subtype evidence 必须是 observed support 的子集，accepted label 也必须来自 A2 candidate 或其 parent label。

## 输出与复现

若提供 `work_dir`，每个 cluster 会写入 `work_dir/traces/<task_id>.json`，其中包含协议/ prompt 版本、初始 evidence packet、selection、质量摘要、结构化最终记录和原始 JSON 响应。COSG 缓存位于 `work_dir/cosg_cache/`，缓存身份绑定 h5ad SHA-256、manifest SHA-256、基因空间、COSG 版本、参数和 marker 快照；不匹配的缓存会被忽略。

benchmark caller `caller/cellclick_v4/cellclick.py` 会额外写入 `protocol_manifest.json` 和 `cellclick_metrics.jsonl`，并把每个任务转换为标准预测 TSV。只有 `status == "validated"` 且 prediction 非空的任务才会进入预测文件；失败任务由上层 runner 记录。

## 相关测试

从仓库根目录运行：

```bash
python -m unittest test.test_cellclick_v4_workflow
python -m unittest test.test_cellclick_v4_a1 test.test_cellclick_v4_a2
python -m py_compile src/cellclick/cellclick/v4/*.py
```

测试使用 mock LLM 和临时 h5ad，不应发起网络请求。修改 prompt、模型 schema、预算或证据策略时，应同时检查 `protocol_fingerprint()` / `contract_fingerprint()` 以及相应的回放和 caller 测试。
