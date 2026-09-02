# EdgeXPU-LLM 设计记录与后续计划

这个文件专门记录 EdgeXPU-LLM 的设计决策、阶段计划、当前限制和后续执行顺序。主 `README.md` 只保留项目介绍、架构和使用方式，避免长期开发过程中信息混在一起。

## 当前阶段判断

当前项目处于 Phase 3。有限 GGUF loader、CPU kernel 和内部 CPU executor 已作为默认 CPU fallback；同模型 `compare` 用于对照 llama bootstrap。

Phase 1 的临时 CPU bootstrap MVP 已经跑通。Phase 2 的 executor/scheduler/trace 闭环已经可验证。Phase 3 已建立最小 native 边界，Qwen2.5 0.5B 的完整 native prefill 与逐步 native decode 已接入；`llama cli` 仅作为无 native session 时的后备。

还不能认为 native EdgeXPU runtime 已完成，因为数值尚未与 llama.cpp 对齐，executor 还是单线程同步执行，尚未替换为 worker queue 或线程池。

## 参考模型 vs 可替换模型包

EdgeXPU-LLM **不是** Qwen2.5-0.5B 专用推理器。Qwen2.5-0.5B GGUF 只是当前的 **reference model pack**：用来把 manifest、executor、native GGUF 路径和验证脚本跑通。用户必须能换成别的 GGUF / 别的架构 / 别的后端 artifact，而不改 runtime 核心。

分层：

- **Runtime 契约**（不随模型变）：manifest、executor job、scheduler、server API、telemetry。
- **模型包**（随模型变）：`model.manifest.json`、artifact 路径、量化格式、可选 chat template。
- **架构适配器**（按 GGUF `general.architecture` 或 manifest `family` 选择）：RoPE 类型、是否 Q/K bias、FFN 形态、tokenizer 类型、chat template。
- **Kernel / 量化**（通用，按模型需要的子集启用）：RMSNorm、matmul、Q4_K 等。
- **设备 backend**（按 artifact `backend` 选择）：CPU native、llama 后备、NPU/dNPU。

当前实现里已经按 GGUF 读 `block_count` / `embedding_length` / KV 头数，层数不是写死 24。native forward 由 `general.architecture` 选择 adapter（当前：`qwen2`、`llama`/`mistral`）：RoPE 类型、Q/K/V bias、FFN=SwiGLU、GPT-2 BPE。chat template 在模型包 manifest（或 GGUF `tokenizer.chat_template` 里带 `{{prompt}}` 的简化模板），runtime 不写死 `<|im_start|>`。换 Phi / Qwen3 仍需新 adapter，而不是改 `native.c` 里的默认分支。

验证脚本和 `examples/models/qwen2.5-0.5b/` 可以继续用这一个包做 CI；禁止把 `qwen2.5`、24 层、`<|im_start|>` 写进 runtime 核心路径。chat template、特殊 token、RoPE 变体属于架构适配器或模型包字段。

后续加模型的正确方式：新增一个 model pack 目录 + manifest。若 `architecture` 已有适配器则应直接跑；没有则加适配器，而不是改 `native.c` 里的 Qwen 分支当默认。

## 已完成记录

### 2026-09-02：项目定位重梳理

- EdgeXPU-LLM 定位为边缘设备离线 LLM runtime，不是普通 llama.cpp wrapper。
- `llama.cpp` 只作为临时 CPU bootstrap，用于验证 MVP 流程。
- 长期目标是自研 native runtime 和 async executor，统一 CPU、NPU、dNPU、memory、flash 的分阶段调度。
- README 先使用中文维护，后续稳定后再转换为英文版本。

### 2026-09-02：MVP 验证状态

- benchmark 已用 Qwen2.5 0.5B GGUF 和新版 `llama cli` 调试通过。
- `/v1/models` 和 non-stream `/v1/chat/completions` 已验证通过。
- streaming 请求样例已补充，server 已按 native token 发送 SSE chunk。
- `scripts/verify_mvp.sh` 已加入，用于一键验证构建、benchmark、server、non-stream、stream 和错误路径。

### 2026-09-02：Phase 2 最小闭环

- 已新增 `include/edgexpu/executor.h` 和 `src/executor.c`。
- runtime 已持有 `edgexpu_executor`。
- executor job 已覆盖 `load_model`、`prepare_prompt`、`tokenize`、`prefill`、`decode_step`、`update_kv_cache`、`prefetch_weights`、`stream_token`、`collect_telemetry`。
- scheduler 已新增 job 级 `edgexpu_schedule_decision`。
- benchmark 已输出 `executor_trace`，每个 job 包含 `backend`、`device`、`scheduler_reason`、`elapsed_seconds` 和 `status`。
- 已新增 `trace` CLI 命令，可直接查看人类可读的 executor trace 表格。
- executor 已支持 `submit -> pending -> run callback -> running -> completed/failed` 的单线程同步执行路径。
- executor job 已可携带 callback 和 user_data，runtime 通过 `submit_runnable -> run_next` 执行 pending job。
- executor 已支持 queue summary、状态计数、空队列错误和非 pending job 重复执行错误检查。
- scheduler policy 已按 job type 拆分：`load_model`、`prepare_prompt`、`tokenize`、`stream_token` 走 CPU，`prefetch_weights` 走 flash，`update_kv_cache` 走 memory，`prefill/decode_step` 根据模型 backend 规划到 CPU/NPU/dNPU。
- scheduler 已预留 device profile 输入，当前可感知 CPU baseline、Rockchip runtime 和 Qualcomm runtime 可用性。
- 已新增 `executor-selftest` CLI 命令，用于验证 executor queue 的基本错误路径。
- 已新增 backend-owned telemetry contract，当前 CPU bootstrap backend 会上报 total/decode seconds、token 近似值、device 和 fallback reason。
- 已新增 `scheduler-selftest` CLI 命令，用于验证 telemetry 输入和 memory/flash policy。
- runtime 已通过 `collect_telemetry` job 保存最近一次 backend telemetry，并在后续 scheduler plan 中作为动态策略输入。

### 2026-09-02：token stream replay

- runtime 已新增 `edgexpu_runtime_generate_stream`。
- batched decode 完成后，按空白/UTF-8 近似 token 提交并执行 `stream_token` job。
- `/v1/chat/completions?stream=true` 先发送 role chunk，再按 token 发送 content delta，最后发送 finish 和 `[DONE]`。
- executor 队列容量提升到 256，并在空间不足时 `drop_terminal` 回收已完成 job。
- `trace` 会打印下一次 decode 的 telemetry-informed plan，无需二次推理。
- `trace` 会在生成结束后打印下一次 decode 的 telemetry-informed plan。
- runtime 已把 batched 生成结果拆成多个 `stream_token` job；server streaming 按近似 token 发送 SSE chunk。
- executor 已支持 `drop_terminal` 回收 completed/failed job，避免长驻 server 把固定容量队列填满。

### 2026-09-02：Phase 3 最小 native 边界

- 新增 GGUF loader，可读取 Qwen2 架构元数据和 tensor catalog，不一次性展开全部权重。
- 新增 native GPT-2 BPE tokenizer，已用 Qwen2.5 0.5B vocab/merges 验证 `Hello EdgeXPU` 可 roundtrip。
- 新增 f32 CPU kernel：`add`、`mul`、`rmsnorm`、`silu`、`softmax`、`matmul`。
- 新增 native KV cache，按 `block_count × head_count_kv × head_dim × max_seq` 分配有限窗口，默认 256。
- runtime 在加载 GGUF 后持有 `edgexpu_native_session`；`tokenize` 走 `cpu.native`，KV job 走 `memory.native`。
- decode 由 native greedy/temperature sampling 逐步完成；`llama cli` 仅在 native session 未就绪时回退。
- 新增 CLI：`inspect-gguf`、`tokenize`、`native-selftest`。

### 2026-09-02：GGUF 权重 + 第一层 native compute

- GGUF 文件通过 mmap 读取；`token_embd.weight` 按 token 做 Q8_0 行反量化。
- 每层权重按需反量化到一块 f32 scratch（Q5_0 / Q8_0 / Q6_K / Q4_K / F32），算完再加载下一层。
- native prefill 已覆盖全部 24 层：embedding → RMSNorm / QKV+bias / Neox RoPE / GQA causal attention / attn output / SwiGLU FFN → `output_norm`。
- 每一层的 K/V 写入 native KV cache；`native-selftest` 校验 `prefill_layers=24`、`decode_tokens` 以及 `layer0_rms` 为有限正值。
- 首次 prefill 会把 24 层权重量化到 f32 并缓存，随后逐步 decode 复用该缓存并按 token 写入 KV。

## 当前限制

- executor 还不是多线程异步队列，目前是单线程同步 callback 执行，但接口已经按 worker queue 方向设计。
- scheduler 已能消费最近一次 backend telemetry，但当前只有提示型策略输入，还没有基于统计窗口、优先级或真实硬件负载做动态调度。
- native tokenizer、KV、prefill 和逐步 decode 已接入自有 kernel；`llama cli` 只作为后备。
- native KV 目前是有上限的 f32 窗口（默认 256 tokens）；prefill 与 decode 都会写入全部层的真实 K/V。
- flash/weight paging 仍是 contract；当前是整层 f32 缓存，而不是按 block 从 flash 调页。
- server streaming 按每个 native token 发送 SSE chunk；尚未与 llama.cpp 做数值对齐。

## 阶段路线图

### Phase 1：Native MVP Baseline

目标：得到一个可构建、可运行、可验证的 native C MVP。

状态：基本完成。

已完成或接近完成：

- C/CMake runtime skeleton
- manifest 加载和 artifact 路径解析
- 临时 `llama.cpp` bootstrap backend
- benchmark、server、request parsing、streaming 样例和 error path 验证
- `scripts/verify_mvp.sh` 一键验证

### Phase 2：Async Executor And Scheduler Trace

目标：把一次推理拆成 scheduler 可理解的 job，并让 benchmark 能输出可追溯 trace。

状态：worker-ready 的单线程同步 job queue、profile-aware scheduler policy、backend telemetry、telemetry-aware scheduler context、token stream replay、trace 输出和 CLI 回归验证已完成。

已完成：

- executor job contract
- job status lifecycle
- scheduler job decision
- runtime load/generate -> executor trace
- benchmark `executor_trace`
- `trace` CLI 调试命令
- 单线程 `run_job` callback 执行路径
- `submit_runnable -> run_next` pending queue 执行路径
- 按 job type 拆分的 scheduler policy
- queue summary 和 `executor-selftest`
- profile-aware scheduler plan API
- backend-owned telemetry contract 和 CPU bootstrap telemetry 输出
- telemetry-aware scheduler context API
- `collect_telemetry` job 保存最近一次 backend telemetry
- `scheduler-selftest`
- telemetry-informed next decode plan
- batched decode 后的 `stream_token` replay 和 SSE chunk
- executor `drop_terminal` 回收 completed/failed job

下一步：

- 评估 Raspberry Pi 部署，并开始把 native decode 与 llama.cpp 做数值/质量对比。

### Phase 3：Native CPU Runtime

目标：减少对 `llama.cpp` shell-out 的依赖，开始建立自有 CPU fallback 路径。

状态：Phase 3 产出已落地。有限 GGUF-derived loader、f32 CPU kernel、内部 CPU executor 路径已作为默认 CPU fallback；`edgexpu compare` 用同一 manifest/prompt 对比 native 与 llama bootstrap。文本不要求逐 token 对齐。

已完成：

- GGUF v3 metadata 和 tensor catalog loader
- Qwen2 GPT-2 BPE tokenizer（vocab + merges）
- f32 CPU kernel：add/mul/rmsnorm/silu/softmax/matmul/linear/RoPE
- native KV cache 所有权（按 architecture 分配有限窗口）
- runtime `tokenize` / `prefill` / `update_kv_cache` job 接入 native session
- mmap GGUF，按层反量化 Q5_0 / Q8_0 / Q6_K / Q4_K 权重
- native 全层 prefill：embedding → 24 层 attention+GQA+RoPE+SwiGLU → output_norm，并写入全部层 KV
- 首次 prefill 缓存 24 层 f32 权重；逐步 decode 用 KV cache + tied embedding logits（greedy / temperature）
- CPU kernel / Q8_0 logits 走 AVX2+FMA；默认 `CMAKE_BUILD_TYPE=Release`（此前空 build type 等于 -O0，compare 会虚高一个数量级）
- runtime 每个生成 token 对应一次 `decode_step` 和一次 `stream_token`
- telemetry 已分开记录 prefill_seconds 与 decode_seconds
- `inspect-gguf`、`tokenize`、`native-selftest`
- 生成请求可选择 `cpu_path`：AUTO / NATIVE / LLAMA_BOOTSTRAP
- `edgexpu compare`：同模型、同 prompt 对比 native CPU fallback 与 llama.cpp shell-out
- architecture adapter：`qwen2`（Neox RoPE + QKV bias）与 `llama`/`mistral`（NORM RoPE，无 QKV bias）
- 模型包 `chat_template`（`{{prompt}}`）；generate 套用，CLI `tokenize` / `native-selftest` 保持原文
- 第二个替换包：`examples/models/smollm2-135m/`

下一步：

- 数值级 logits 与 llama.cpp 对齐（按 adapter 算子，不是 Qwen 专用分支）。
- Raspberry Pi ARM fallback 验证。

### Phase 4：NPU / dNPU Backend Adapter

目标：接入真实边缘硬件 runtime，让 scheduler 能看到不同设备的真实能力。

优先级：

1. Rockchip RKLLM/RKNN
2. Qualcomm QNN
3. NXP / MediaTek / Intel OpenVINO
4. Sophgo / Axera / Horizon
5. Hailo / Coral 作为轻量任务或视觉/embedding 辅助

验证重点：

- transformer operator 支持
- quantization 格式
- prefill/decode 分项性能
- dynamic shape 或固定 shape 编排能力
- KV cache 管理方式
- vendor runtime 离线部署能力
- telemetry 完整度

### Phase 5：Stage-Aware Scheduler

目标：让 scheduler 根据 job type、设备能力、runtime telemetry 和 workload priority 做调度。

计划：

- 按设备能力路由 prefill、decode、verification 和 background job。
- 加入 prompt cache 和 KV cache policy。
- 重叠 CPU preprocessing、accelerator execution、flash prefetch 和 streaming output。
- 为本地 agent workload 加入 foreground/background priority。

### Phase 6：Edge Optimization And Research

目标：处理真实边缘设备上的内存、flash、长上下文和 agent workload 压力。

计划：

- flash-aware weight paging 和 prefetch
- speculative decoding
- activation profiling
- hot/cold placement
- sparse FFN / sparse attention
- MoE routing
- 真实边缘硬件长时运行测试

## 近期执行计划

后续开发按这个顺序推进：

1. ~~把 native forward 收成 architecture adapter：由 GGUF `general.architecture` 描述 RoPE / bias / FFN / tokenizer，Qwen2 只是第一个适配器实现。~~ 已做：`include/edgexpu/arch.h`，Qwen2 与 Llama/Mistral adapter。
2. ~~模型包补齐 chat template（manifest 或 GGUF `tokenizer.chat_template`），runtime 不写死 chat 标记。~~ 已做：manifest `chat_template` + `{{prompt}}` 替换；只在 generate 路径套用。
3. ~~用第二个不同架构的小 GGUF 做替换验证：同一套 CLI/server，只换 manifest。~~ 已做：`examples/models/smollm2-135m/`（llama + GPT-2 BPE + NORM RoPE，无 QKV bias）。
4. 数值级与 llama.cpp 对齐（adapter 描述的算子，不是 Qwen 专用分支）、logits 加速、Raspberry Pi ARM fallback。
5. 在模型可替换且 CPU path 可对比之后，再接 Rockchip / Qualcomm / dNPU backend。

已实现的 trace 调试命令（当前示例包，不是唯一支持的模型）：

```bash
./build/edgexpu trace examples/models/qwen2.5-0.5b/model.manifest.json "Explain EdgeXPU-LLM briefly."
```

## 验证命令

完整 MVP 回归：

```bash
bash scripts/verify_mvp.sh
```

单独 benchmark：

```bash
./build/edgexpu benchmark examples/models/qwen2.5-0.5b/model.manifest.json "Explain EdgeXPU-LLM briefly."
```

同模型 CPU fallback 对比：

```bash
./build/edgexpu compare examples/models/qwen2.5-0.5b/model.manifest.json "Explain EdgeXPU-LLM briefly."
```

executor 自测：

```bash
./build/edgexpu executor-selftest
```

native 自测：

```bash
./build/edgexpu native-selftest examples/models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

单独 server 验证：

```bash
./build/edgexpu serve examples/models/qwen2.5-0.5b/model.manifest.json 8000
```

```bash
curl http://127.0.0.1:8000/v1/models
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @examples/requests/chat_completion.json
```

## 追溯规则

- 重要方向变更写入本文件。
- Runtime 必须可替换模型；Qwen2.5-0.5B 只是 reference pack，禁止把单一模型的 chat/RoPE/层数写进核心路径。
- 已验证的命令写入本文件。
- 临时方案必须标注“临时”以及未来替换路径。
- 新增 backend 前先补充调度策略和验收标准。
- README 主文件只写项目介绍和使用方式，详细阶段计划放在本文件。
