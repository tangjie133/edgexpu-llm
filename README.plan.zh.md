# EdgeXPU-LLM 设计记录与后续计划

这个文件专门记录 EdgeXPU-LLM 的设计决策、阶段计划、当前限制和后续执行顺序。主 `README.md` 只保留项目介绍、架构和使用方式，避免长期开发过程中信息混在一起。

## 当前阶段判断

当前项目处于 Phase 3：**功能闭环已完成；3.1 公平测速已完成；3.2 decode 已过 0.6× 停手线；3.3 ARM 交叉编译与 NEON 单元测试已可在 x86 上用 qemu 跑。** 默认 `cpu.native`；`llama cli` 仅作无 native session 时的后备，以及可选对照。

还不能认为 native runtime 已完成：prefill 仍约 0.5× llama CPU；executor 仍是单线程同步；树莓派实机 greedy 锁点尚未验收；更长 prompt / 更大 GGUF 的板上内存还没压过。greedy n=4 的 prompt 在 `scripts/verify.locks`；各包 id 在 `examples/models/<pack>/verify.lock`。

余下 Phase 3 工作是 3.3 树莓派实机：NEON 构建 + 参考包 greedy 锁点 + 不 OOM / mmap。decode 不再为追 llama 改 kernel。

## 参考模型 vs 可替换模型包

EdgeXPU-LLM **不是**某一个 GGUF 专用推理器。当前仓库里的 **native 参考包** 是 `examples/models/smollm2-135m/`。用户必须能换成别的 GGUF / 别的架构 / 别的后端 artifact，而不改 runtime 核心。

分层：

- **Runtime 契约**（不随模型变）：manifest、executor job、scheduler、server API、telemetry。
- **模型包**（随模型变）：`model.manifest.json`、artifact 路径、量化格式、可选 chat template。
- **架构适配器**（按 GGUF `general.architecture` 或 manifest `family` 选择）：RoPE 类型、是否 Q/K bias、FFN 形态、tokenizer 类型、chat template。
- **Kernel / 量化**（通用，按模型需要的子集启用）：RMSNorm、matmul、Q4_K 等。
- **设备 backend**（按 artifact `backend` 选择）：产品路径 `cpu.native`；NPU/dNPU；llama 仅 `compare`。

当前实现里已经按 GGUF 读 `block_count` / `embedding_length` / KV 头数，层数不是写死 24。native forward 由 `general.architecture` 选择 adapter（当前：`qwen2`、`llama`/`mistral`）：RoPE 类型、Q/K/V bias、FFN=SwiGLU、GPT-2 BPE。chat template 在模型包 manifest，支持 `{{prompt}}` / `{{system}}` / `{{#system}}` / `{{#message}}`。runtime 不写死 `<|im_start|>`。换 Phi / Qwen3 仍需新 adapter，而不是改 `native.c` 里的默认分支。

验证脚本扫描 `examples/models/*/verify.lock`；禁止把 `qwen2.5`、24 层、`<|im_start|>` 写进 runtime 核心路径。chat template、特殊 token、RoPE 变体属于架构适配器或模型包字段。

后续加模型：复制 `examples/models/_template/` 与 `src/arch/_template.c`，见 `CONTRIBUTING.md`。禁止改 `native.c` 层循环当默认模型。

### 2026-09-03：模块边界（开源贡献路径）

- 架构插件：`src/arch/*.c` + `edgexpu_arch_register`；tensor 名模板离开 `native.c` 硬编码后缀。
- Backend 分步 vtable：`load` / `tokenize` / `ensure_window` / `prefill` / `decode_step`；`cpu.native` 与 llama `generate` 分开。
- Scheduler 资源预算：权重 mmap + KV + prefill scratch，超过设备 RAM 85% 拒绝 native load。
- `CONTRIBUTING.md`、`examples/models/_template/`、`src/arch/_template.c`。
- Qwen3.5 hybrid 已登记 plugin，SSM 前向未实现：产品仍声明 `cpu.native`，load/generate 失败而不是改走 llama。CI `NATIVE=0` 只锁 tokenize。
- `verify_arm.sh` 的 pack 元数据从 JSON 读路径，交叉编译的 aarch64 `edgexpu` 不再被裸 exec；`EDGEXPU_ARM_FULL=1` 在有 native GGUF 却 0 次 dump-logits 时失败，禁止 skip 当通过。

## 已完成记录

### 2026-09-03：测试记录第 2 轮 P2 修补

- Next Decode Plan：`cpu.native` 时 policy 为 `telemetry_keep_native`（baseline 仍为 `telemetry_keep_cpu`）。
- `benchmark` JSON 输出 `finish_reason`；新增 `edgexpu generate` 打纯文本。
- `copy_replay_token` / `command -v` 去掉 format-truncation。
- qemu 跑 ARM 时设 `EDGEXPU_EMULATED=1`，`capabilities` 带 `"emulated": true`；核数/内存不当板级结果。
- 参考包标明 `name=Qwen2.5-Coder-0.5B-Instruct`，`family=qwen2.5-coder`，`model_id` 仍为 CI 用的 `qwen2.5-0.5b`。
- artifact `backend` 改为 `cpu.native`；`fallback_policy` 仍是 llama bootstrap。
- HTTP 按 `Content-Length` 拼包，超缓冲返回 413。

### 2026-09-03：HTTP / telemetry 契约修补

- HTTP `/v1/chat/completions` 解析完整 `messages`（system + 多轮），并校验 `model` 与已加载 manifest。
- 触达 `max_tokens` 时 `finish_reason` 为 `length`，EOS 为 `stop`。
- native prefill/decode 在 trace 标 `cpu.native`；Next Decode Plan 带上 native_ready，不再误报 cpu.baseline。
- `memory_used_mb` = mmap 权重 + KV（参考包约 385MB），不再只计 KV。
- generate/HTTP 缓冲 `EDGEXPU_TEXT_PROMPT`（32KiB）；chat 模板支持 `{{#system}}` / `{{#message}}`。
- `prefetch_weights` 对 mmap 做 `madvise(WILLNEED)`；`update_kv_cache` 在 native 路径校验已写入的窗口，不再空跑 extend。flash paging 仍未做。

### 2026-09-03：Phase 3.3 ARM 入口（交叉编译 + qemu）

- `cpu_kernel` 的 f32 `dot` / `saxpy` / `linear` 在 ARM 走 NEON（与量化点积同一套 `EDGEXPU_NEON`）。
- `capabilities` 增加 `simd`（`avx2` / `neon` / `scalar`）。
- 交叉编译：`cmake/aarch64-linux-gnu.cmake`，默认 `-march=armv8-a`（Pi 4，无 DOTPROD）。Pi 5 用 `-DEDGEXPU_ARM_DOTPROD=ON`。
- `scripts/verify_arm.sh`：x86 上交叉编译 + qemu 跑 unit（`simd=neon`）；实机或 `EDGEXPU_ARM_FULL=1` 再跑参考包 greedy 锁点。
- `native-selftest` 打印 `simd=`；加载 GGUF 后断言 `prompt + context_length` 超窗报错（`kv_overflow=ok`）。
- 实机 Pi 尚未跑 greedy 锁点；3.3 未关门。

### 2026-09-03：Phase 3.1 公平测速

- 新增 `scripts/bench_cpu.sh`；`edgexpu benchmark` 可传 `max_tokens`，tok/s 用分项时间。
- 本机 n=32：native decode 52.4 tok/s vs llama CPU 45.2 tok/s（1.16×）；prefill 0.49×。
- llama 必须 `CUDA_VISIBLE_DEVICES=`，否则 `-ngl 0` 仍可能初始化 CUDA。
- 按停手线关闭 3.2 kernel。

### 2026-09-03：Phase 3 优化计划收口

- Phase 3 拆成 3.0 功能（已完成）、3.1 公平测速、3.2 kernel 到 0.6× 停手、3.3 Pi 正确性。
- 明确不做：ggml 级 matmul 图、异步 executor、flash paging、为打平 llama 改架构。
- decode 已过 0.6× 则停手；prefill 剩余差距不单独开阶段。

### 2026-09-02：architecture adapter 与可替换模型包

- native forward 不再写死 Qwen2：`src/arch.c` 按 GGUF `general.architecture` 选择 RoPE / QKV bias / FFN / tokenizer。
- 当前 adapter：`qwen2`（Neox + bias）、`llama`/`mistral`（NORM，无 bias）。
- 模型包 `chat_template` 用 `{{prompt}}`；只在 generate 路径套用。runtime 不写死 `<|im_start|>`。
- GPT-2 bytes 编解码与 `<...>` 特殊 token 整段 lookup。参考包 generate 已能输出连贯英文。
- 第二个包 `examples/models/smollm2-135m/`：同一套 CLI，只换 manifest；adapter=llama。

### 2026-09-02：验证入口收拢

- 日常只跑 `scripts/verify_mvp.sh`；greedy n=4 的 prompt 在 `scripts/verify.locks`，各包 id 在 `examples/models/<pack>/verify.lock`。
- `dump-logits` 的 piece / greedy_text 转义换行，避免看起来像空 piece。
- 对照 llama.cpp 用可选的 `scripts/align_llama.sh`；默认 `verify_mvp.sh` 不调用 llama。

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
- decode 由 native greedy/temperature sampling 逐步完成；产品路径不回退 `llama cli`。
- 新增 CLI：`inspect-gguf`、`tokenize`、`native-selftest`。

### 2026-09-02：GGUF 权重 + 第一层 native compute

- GGUF 文件通过 mmap 读取；`token_embd.weight` 按 token 做 Q8_0 行反量化。
- 每层权重按需反量化到一块 f32 scratch（Q5_0 / Q8_0 / Q6_K / Q4_K / F32），算完再加载下一层。
- native prefill 已覆盖全部 24 层：embedding → RMSNorm / QKV+bias / Neox RoPE / GQA causal attention / attn output / SwiGLU FFN → `output_norm`。
- 每一层的 K/V 写入 native KV cache；`native-selftest` 校验 `prefill_layers=24`、`decode_tokens` 以及 `layer0_rms` 为有限正值。
- 首次 prefill 绑定各层 mmap 量化权重指针，decode 按行 fused 点积，不把整层展开成 f32。

## 当前限制

- executor 还不是多线程异步队列，目前是单线程同步 callback 执行，但接口已经按 worker queue 方向设计。
- scheduler 已能消费最近一次 backend telemetry，但当前只有提示型策略输入，还没有基于统计窗口、优先级或真实硬件负载做动态调度。
- native tokenizer、KV、prefill 和逐步 decode 已接入自有 kernel；`llama cli` 只作为后备。
- native KV 按 `prompt + max_tokens` 分配（load 占位 256），超过模型 `context_length` 报错，不再截断。
- 层权重保持 GGUF 量化（Q4_K/Q5_K/Q4_0/Q6_K/Q5_0/Q8_0/F16/F32），flash paging 仍是 contract。
- server streaming 按每个 native token 发送 SSE chunk；greedy 前 4 个 token 按模型包 `verify.lock` 锁。
- CPU fallback decode 已过桌面 0.6× 停手线（3.1 n=32：native 52.4 / llama CPU 45.2 ≈ 1.16×）。prefill 仍约 0.5×，不为追 GEMM 开阶段。

## CPU fallback 验收标准

两根尺子分开，禁止混成一根 tok/s 排名。

### 公平对照（强制）

和 llama.cpp 比速度时必须同时满足：

- 同一份 GGUF、同一 prompt、greedy（`temperature=0`）
- llama 关加速器：`llama-cli --n-gpu-layers 0`（或等价 CPU-only）
- 分项记录 prefill tok/s 与 decode tok/s，不拿 wall-clock 里的 load 当 decode
- **GPU / NPU 上的 llama 数字不是 native CPU KPI**

`edgexpu compare` 若未关 GPU，其耗时不能当验收。compare 在 native 不能 load 时仍会跑 llama 腿（例如 Qwen3.5 缺 SSM adapter）；产品 `generate` 不会。数值对齐仍用 `scripts/align_llama.sh`（`--no-conversation`）。

### 尺子 A：桌面 CPU fallback（必须够用，不必赢）

平台：x86 Linux，native 参考包 SmolLM2-135M。

| 项 | 标准 |
| --- | --- |
| Phase 3 完成 | 默认 `cpu.native`、`verify_mvp.sh` 不 shell-out llama、greedy n=4 锁点 |
| decode 下限 | native ≥ **0.6×** 同机 llama CPU（3.1 已达 1.16×，3.2 停手） |
| decode 拉伸 | 0.8×；到此停止为追 ggml 开新阶段 |
| 打平 llama CPU | **不是** 完成条件，也不是接 NPU 的前置 |

未到 0.6× 时只收紧最热 GEMV / 整数 SIMD（Phase 3.2），不改架构图。3.1 已达 1.16×，3.2 停手。prefill 剩余差距记为已知限制。

### 尺子 B：Raspberry Pi / ARM（正确性与资源，不是决赛场）

树莓派用来验 ARM/NEON、内存和存储 mmap，**不设相对 llama 的 tok/s 达标线**。

通过：

- NEON 构建可跑，参考包 greedy 锁点仍成立
- 参考包推理不 OOM（按 `prompt + max_tokens` 扩 KV，超窗报错）
- 从板载存储 mmap GGUF 可完成 prefill/decode

记录 tok/s 只作 telemetry。相对 llama 若从桌面 ~2× 变成 ~4×，先查 NEON / 线程 / SD mmap，不因此推迟 NPU。

产品 tok/s 验收在 Rockchip / Qualcomm 板子上，不在 Pi CPU 上。

### 接 NPU 的条件

Phase 4 开工条件是 CPU path **可对比**，不是 native 打平 llama：

- 默认推理走 `cpu.native`
- greedy 锁点绿
- 存在一次公平 CPU-only llama 对照记录（即使尚未到 0.6×）

### 2026-09-03：CPU 差距写成硬标准

- 电脑上相对 llama CPU ~2× 说明 fallback 引擎未完成，不说明框架失败；与 5090 GPU llama 对比无效。
- 桌面 decode 目标 ≥0.6× llama CPU，拉伸 0.8× 后停手。
- 树莓派不设 tok/s 线；NPU 排在「可对比」之后，不是「打平 llama」之后。

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

- ARM 正确性见 Phase 3.3，不在 Phase 2 重复排期。

### Phase 3：Native CPU Runtime

目标：减少对 `llama.cpp` shell-out 的依赖，建立自有 CPU fallback。**不是**做成第二个 ggml。

状态：3.0 功能闭环已落地。3.1 公平测速已完成。3.2 decode 已过 0.6×，kernel 停手。3.3 已有 aarch64 交叉编译、NEON 单元测试和 qemu 入口；树莓派实机 greedy 锁点未跑。

已完成（3.0 功能）：

- GGUF v3 metadata 和 tensor catalog loader
- GPT-2 BPE tokenizer；architecture adapter（`qwen2` / `llama` / `mistral`）
- mmap 量化权重 + fused / Q8 整数点积；native KV；全层 prefill 与逐步 decode
- greedy / temperature / top_p；greedy n=4 锁点
- OpenMP + 首次 GEMV 探线程；层 GEMV 与 logits 可走 Q8_0 / Q8_K
- `inspect-gguf` / `tokenize` / `dump-logits` / `native-selftest` / `benchmark` / `trace`
- 第二个替换包 `examples/models/smollm2-135m/`

本机参考包公平 CPU 对照（2026-09-03，`scripts/bench_cpu.sh`，n=32，llama `-ngl 0` 且 `CUDA_VISIBLE_DEVICES=`）：

| 项 | native | llama CPU | 比值 |
| --- | --- | --- | --- |
| decode tok/s | 52.4 | 45.2 | **1.16×** |
| prefill tok/s | 115.4 | 234.5 | 0.49× |

decode 已过 0.6× 停手线（并高于本次 llama CPU）。**Phase 3.2 kernel 不再开新项。** prefill 仍约 0.5×，记为已知限制，不为追 ggml GEMM 开阶段。短 `trace` n=4 的 ~24 tok/s 不作 KPI。

#### 3.1 公平测速（已完成）

入口：`bash scripts/bench_cpu.sh`（不进 `verify_mvp.sh`）。

- 同一 GGUF、greedy，decode n=32
- llama `-ngl 0` 且 `CUDA_VISIBLE_DEVICES=`（仅 `-ngl 0` 仍会初始化 CUDA，不公平）
- tok/s 用分项时间；新版 llama-cli 解析 `[ Prompt: … t/s | Generation: … t/s ]`
- 2026-09-03 本机：decode **1.16×** llama CPU，prefill **0.49×**

短 `trace` n=4 不作 KPI。native generate 会套 chat template，llama 用原文 prompt，prefill token 数可能不同。

#### 3.2 桌面 kernel（停手）

decode 已 ≥0.6×，按计划停手。不再为打平 llama 或补 prefill GEMM 开 kernel 阶段。prefill 0.49× 留在「当前限制」。

若日后回归到 decode <0.6×，再按热路径点积排查，不提前开新阶段。

本阶段明确不做：

- 权重 repack / 模仿 ggml 的完整 mat-mul 图
- 异步多线程 executor（那是 Phase 2/5 的队列演进）
- flash paging、KV 量化、speculative decode
- 为打平 llama 重写架构或引入 BLAS

#### 3.3 Raspberry Pi / ARM（正确性，不挡 NPU）

与 3.2 可并行，不依赖 0.6×。桌面可先交叉编译验证 NEON，不替代实机。

- NEON 构建 + 参考包 greedy 锁点
- 参考包不 OOM；超窗报错
- 板载存储 mmap 能跑完 prefill/decode
- tok/s 只记录；无相对 llama 达标线

入口：`bash scripts/verify_arm.sh`（不进 `verify_mvp.sh`）。

x86 开发机：安装 `gcc-aarch64-linux-gnu` 与 `qemu-user-static`，脚本交叉编译 Pi 4 基线并 qemu 跑 kernel/KV/executor。`EDGEXPU_ARM_FULL=1` 才在 qemu 里跑 GGUF greedy（慢）。

板上：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# Pi 5:
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DEDGEXPU_ARM_DOTPROD=ON
bash scripts/verify_arm.sh
```

#### Phase 3 出口

| 项 | 是否必须 |
| --- | --- |
| 默认 native、锁点、verify 不 shell-out | 是（已满足） |
| 一次公平 CPU-only 对照记录 | 是（3.1） |
| decode ≥0.6× llama CPU | 是（3.1 已达 1.16×，3.2 停手） |
| Pi NEON / 内存 | 要做，不挡 Phase 4 |
| 打平 llama CPU | 否 |

### Phase 4：NPU / dNPU Backend Adapter

目标：接入真实边缘硬件 runtime，让 scheduler 能看到不同设备的真实能力。

开工条件：CPU path **可对比**（默认 `cpu.native`、greedy 锁点、存在公平 CPU-only llama 对照），**不是** native tok/s 打平 llama.cpp。产品 tok/s 验收在 NPU 板子上，不在树莓派 CPU 上。

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
4. ~~数值级与 llama.cpp 对齐、logits 加速~~ 已做：greedy n=4 按包锁进 `verify.lock`；量化 fused / Q8 点积；KV 按请求扩窗；top_p。
5. ~~**Phase 3.1** 公平 CPU 测速~~ 已做：`scripts/bench_cpu.sh`；decode 1.16× llama CPU，prefill 0.49×。
6. ~~**Phase 3.2** kernel 到 0.6×~~ 停手：decode 已过线，不再开 kernel 项。
7. **Phase 3.3**：树莓派 NEON / 不 OOM / mmap。不设相对 llama 的 tok/s 线。桌面入口 `scripts/verify_arm.sh`；实机 greedy 锁点仍待跑。
8. CPU path 已可对比，可接 Rockchip / Qualcomm / dNPU。产品 tok/s 在 NPU 板上验收。

已实现的 trace 调试命令（当前示例包，不是唯一支持的模型）走 `edgexpu` 无参数帮助，不在此重复罗列。

## 验证

只维护下列入口，锁点不写进文档正文：

| 入口 | 用途 |
| --- | --- |
| `bash scripts/verify_mvp.sh` | 日常 / CI：只测 native CPU fallback，不 shell-out llama.cpp |
| `scripts/verify.locks` | 共享 dump prompt 与 greedy n |
| `examples/models/<pack>/verify.lock` | 该包 tokenize / greedy id；缺该包 GGUF 则跳过该包；零个 `NATIVE=1` 包则 `verify_mvp.sh` 失败 |
| `bash scripts/align_llama.sh` | 可选对照 llama.cpp 数值（`--no-conversation` / greedy id） |
| `bash scripts/bench_cpu.sh` | 可选公平 CPU 速度对照（n=32，llama `-ngl 0` + 隐藏 GPU） |
| `bash scripts/verify_arm.sh` | Phase 3.3：aarch64 NEON 构建；x86 上 qemu unit，实机跑 greedy 锁点 |

改数值锁点改对应包的 `verify.lock`。速度达标线不进 CI，避免 GPU llama 或短序列噪声误杀。

## 追溯规则

- 重要方向变更写入本文件。
- Runtime 必须可替换模型；禁止把单一模型的 chat/RoPE/层数写进核心路径。
- 追上 llama.cpp CPU tok/s 不是阶段完成条件；CPU fallback 与边缘产品用两根尺子，见「CPU fallback 验收标准」。
- 已验证的命令写入本文件。
- 临时方案必须标注“临时”以及未来替换路径。
- 新增 backend 前先补充调度策略和验收标准。
- README 主文件只写项目介绍和使用方式，详细阶段计划放在本文件。
