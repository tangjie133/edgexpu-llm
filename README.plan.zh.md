# EdgeXPU-LLM 设计记录与后续计划

这个文件专门记录 EdgeXPU-LLM 的设计决策、阶段计划、当前限制和后续执行顺序。主 `README.md` 只保留项目介绍、架构和使用方式，避免长期开发过程中信息混在一起。

## 当前阶段判断

当前项目处于 Phase 2 早期。

Phase 1 的临时 CPU bootstrap MVP 已经跑通，可以作为后续回归基线。Phase 2 已完成可验证闭环：scheduler 可以结合 manifest 和 device profile 为 job 生成决策，runtime 可以通过单线程 executor queue 执行 job，benchmark 和 trace 可以输出可验证的 `executor_trace`。

还不能认为 native EdgeXPU runtime 已完成，因为目前推理仍依赖临时 `llama cli`，executor 还是单线程同步执行，尚未替换为 worker queue 或线程池。

## 已完成记录

### 2026-09-02：项目定位重梳理

- EdgeXPU-LLM 定位为边缘设备离线 LLM runtime，不是普通 llama.cpp wrapper。
- `llama.cpp` 只作为临时 CPU bootstrap，用于验证 MVP 流程。
- 长期目标是自研 native runtime 和 async executor，统一 CPU、NPU、dNPU、memory、flash 的分阶段调度。
- README 先使用中文维护，后续稳定后再转换为英文版本。

### 2026-09-02：MVP 验证状态

- benchmark 已用 Qwen2.5 0.5B GGUF 和新版 `llama cli` 调试通过。
- `/v1/models` 和 non-stream `/v1/chat/completions` 已验证通过。
- streaming 请求样例已补充，目前是先生成再返回 SSE chunk，不是真正 token-by-token streaming。
- `scripts/verify_mvp.sh` 已加入，用于一键验证构建、benchmark、server、non-stream、stream 和错误路径。

### 2026-09-02：Phase 2 最小闭环

- 已新增 `include/edgexpu/executor.h` 和 `src/executor.c`。
- runtime 已持有 `edgexpu_executor`。
- executor job 已覆盖 `load_model`、`prepare_prompt`、`prefill`、`decode_step`、`update_kv_cache`、`prefetch_weights`、`stream_token`、`collect_telemetry`。
- scheduler 已新增 job 级 `edgexpu_schedule_decision`。
- benchmark 已输出 `executor_trace`，每个 job 包含 `backend`、`device`、`scheduler_reason`、`elapsed_seconds` 和 `status`。
- 已新增 `trace` CLI 命令，可直接查看人类可读的 executor trace 表格。
- executor 已支持 `submit -> pending -> run callback -> running -> completed/failed` 的单线程同步执行路径。
- executor job 已可携带 callback 和 user_data，runtime 通过 `submit_runnable -> run_next` 执行 pending job。
- executor 已支持 queue summary、状态计数、空队列错误和非 pending job 重复执行错误检查。
- scheduler policy 已按 job type 拆分：`load_model`、`prepare_prompt`、`stream_token` 走 CPU，`prefetch_weights` 走 flash，`update_kv_cache` 走 memory，`prefill/decode_step` 根据模型 backend 规划到 CPU/NPU/dNPU。
- scheduler 已预留 device profile 输入，当前可感知 CPU baseline、Rockchip runtime 和 Qualcomm runtime 可用性。
- 已新增 `executor-selftest` CLI 命令，用于验证 executor queue 的基本错误路径。

## 当前限制

- executor 还不是多线程异步队列，目前是单线程同步 callback 执行，但接口已经按 worker queue 方向设计。
- scheduler 已有 job type policy 和 device profile 输入，但还没有基于运行时 telemetry 的动态策略。
- native CPU runtime 尚未开始，当前推理仍依赖临时 `llama cli`。
- KV cache、memory、flash job 目前只是 contract 记录，还未接入真实资源管理。
- server streaming 还不是 token-by-token streaming。

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

状态：worker-ready 的单线程同步 job queue、profile-aware scheduler policy、trace 输出和 CLI 回归验证已完成。

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

下一步：

- 引入 backend-owned telemetry，让 scheduler 不只依赖静态 profile。
- 设计真正 token-by-token streaming 的 executor job 流程。
- 准备 Phase 3 native CPU runtime 的最小 loader/kernel 边界。

### Phase 3：Native CPU Runtime

目标：减少对 `llama.cpp` shell-out 的依赖，开始建立自有 CPU fallback 路径。

计划：

- 为有限 GGUF-derived 路径加入 native model loading。
- 实现第一批 CPU fallback kernel，或接入最小内部 CPU executor。
- 把 benchmark timing 和 telemetry 下沉到 backend 自己上报。
- 用同一个模型对比 `llama cli` bootstrap 和 native CPU path。

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

1. 为 backend-owned telemetry 定义数据结构，记录 prefill/decode latency、token rate、memory 和 fallback reason。
2. 完成真正 token-by-token streaming 的 executor job 流程设计。
3. 评估 Raspberry Pi 部署，作为 ARM CPU fallback 的可移植性验证。
4. 准备 Phase 3 native CPU runtime 的最小边界：loader、tokenizer、CPU kernel、KV cache ownership。
5. 在 executor、scheduler 和 telemetry 稳定后，再开始 Rockchip / Qualcomm / dNPU backend adapter。

已实现的 trace 调试命令：

```bash
./build/edgexpu trace examples/models/qwen2.5-0.5b/model.manifest.json "Explain EdgeXPU-LLM briefly."
```

当前推荐下一步是定义 backend-owned telemetry 数据结构。

## 验证命令

完整 MVP 回归：

```bash
bash scripts/verify_mvp.sh
```

单独 benchmark：

```bash
./build/edgexpu benchmark examples/models/qwen2.5-0.5b/model.manifest.json "Explain EdgeXPU-LLM briefly."
```

executor 自测：

```bash
./build/edgexpu executor-selftest
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
- 已验证的命令写入本文件。
- 临时方案必须标注“临时”以及未来替换路径。
- 新增 backend 前先补充调度策略和验收标准。
- README 主文件只写项目介绍和使用方式，详细阶段计划放在本文件。
