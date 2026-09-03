# EdgeXPU-LLM

EdgeXPU-LLM 是一个面向边缘设备的离线 LLM 推理运行时。它的目标是在同一套本地运行时契约下，统一 ARM CPU fallback、Rockchip NPU、Qualcomm NPU，以及后续可能加入的 dNPU 辅助执行路径。

这个项目不是把云端加速 serving 框架缩小后搬到小设备上。它要解决的是边缘设备自己的问题：NPU 图限制、CPU/NPU/dNPU 协同、统一内存压力、flash 带宽、量化格式、KV cache 增长、散热功耗限制，以及完全本地的隐私敏感工作负载。

第一阶段不覆盖 STM32 这类微控制器。当前目标设备是能运行 Linux 的边缘平台，例如 Raspberry Pi、RK3588/RK3576 开发板，以及 Qualcomm Snapdragon / Dragonwing 平台。

## 项目边界

`llama.cpp` 只是临时方案。

当前 MVP 使用 `llama.cpp` 或兼容工具作为 CPU bootstrap backend，目的是先验证 CLI、模型 manifest、benchmark 输出、本地 API 形态和基础 runtime 流程。在 native executor 成熟之前，它是一个方便的临时执行后端。

长期目标是用 EdgeXPU 自己的运行时替代 shell-out 推理，逐步实现：

- 原生模型加载和 artifact 管理
- CPU operator 执行和 fallback kernel
- NPU/dNPU backend adapter
- 异步 load、tokenize、prefill、decode、KV cache、memory、flash、stream job
- CPU、NPU、dNPU、memory、storage 之间的分阶段调度
- backend 无关的 telemetry 和 benchmark trace

如果项目最终只是把 `llama.cpp` 包成 OpenAI-compatible API，那它只是一个 baseline adapter，不是 EdgeXPU-LLM 的核心。EdgeXPU-LLM 的核心价值应该来自 native runtime contract、async executor、设备能力 profiling、scheduler、cache policy、memory policy，以及未来的 accelerator routing。

## 当前状态

当前仓库已经有一套 native C MVP 骨架：

- 基于 CMake 的 C runtime library 和 CLI
- 模型 manifest loader
- 设备能力 profiler
- 简单 backend selector
- Phase 2 async executor contract、单线程 job queue 和 benchmark trace 最小闭环
- Phase 3 Native CPU Runtime：有限 GGUF loader、CPU kernel、内部 executor；默认走 native CPU fallback，`compare` 与 llama bootstrap 同模型对比
- 通过 `llama`、`llama-cli`、`powerinfer` 或 `main` 调用的临时 CPU baseline backend
- 带 stage trace 和 executor trace 的 benchmark 命令
- 带 queue summary 和 scheduler policy 的 `trace` 调试命令
- `executor-selftest`、`scheduler-selftest`、`native-selftest` 自测命令
- `inspect-gguf` 和 `tokenize` 调试命令
- backend-owned telemetry 第一版，并已作为下一次 scheduler plan 的输入
- 本地 OpenAI-compatible `/v1/models` 和 `/v1/chat/completions` server
- streaming 请求会按每个 native token 发送 SSE chunk
- Qwen2.5 0.5B 参考模型包，以及 SmolLM2-135M 替换验证包（只换 manifest）
- GGUF `general.architecture` 插件（`src/arch/`：qwen2、llama、qwen35 识别）
- 模型包 `chat_template`，runtime 不写死 chat 标记
- 贡献入口 `CONTRIBUTING.md` 与 `examples/models/_template/`
- MVP 验证脚本 `scripts/verify_mvp.sh`

本地 benchmark 路径已经用 GGUF 模型和 native CPU prefill/decode 验证通过。runtime 通过单线程 runnable executor queue 执行 load、tokenize、prefill、decode、KV cache、stream 和 telemetry job。`tokenize` 走 native GGUF BPE tokenizer；`prefill` 跑完全部 transformer 层并写入 KV；每个生成 token 对应一次 native `decode_step` 和一次 `stream_token`。`llama cli` 仅在 native session 未就绪时作为后备。benchmark 会输出 `backend_telemetry`（含 prefill/decode 分项时间）、queue summary、scheduler policy 和 `executor_trace`。

## 总体架构

```text
Application Layer
  Local Chat / IDE Agent / RAG / Tool Calling
        |
Unified Local API
  OpenAI-compatible HTTP API
  generate / streamGenerate / embed / classify / toolCall
        |
EdgeXPU Runtime Core
  Model Manager
  Capability Profiler
  Async Executor
  Stage-Aware Scheduler
  KV Cache Manager
  Memory and Flash Manager
  Security Manager
        |
Backend Adapter Layer
  Native CPU Backend
  Rockchip RKLLM/RKNN Backend
  Qualcomm QNN Backend
  ONNX Runtime GenAI Backend
  ExecuTorch Backend
  Temporary llama.cpp Bootstrap Backend
        |
Hardware Layer
  ARM CPU
  Rockchip NPU
  Qualcomm Hexagon NPU
  dNPU
  DRAM / Unified Memory / Flash
```

## 执行模型

运行时不应该把一次 LLM 推理当成一个黑盒 backend call。LLM 推理的不同阶段有不同的硬件特征。

Prefill：

- 并行处理 prompt
- 计算密集
- 如果图和 shape 支持，通常更适合 NPU/dNPU

Decode：

- 每次生成一个 token
- 经常受 memory bandwidth 限制
- 因为动态 shape 和调度开销，不一定总是适合 NPU

KV cache 和 memory：

- 随 context length 和 session 数量增长
- 需要 placement、eviction、reuse 和量化策略
- 往往是边缘设备上最容易爆掉的资源

Flash 和模型驻留：

- 当权重不能轻松全部放进内存时很重要
- 后续需要支持 prefetch、paging、hot/cold placement

长期的 internal executor 应该把推理拆成类似这样的 job：

```text
loadModelArtifact
preparePrompt
tokenize
prefill
decodeStep
updateKvCache
prefetchWeights
streamToken
collectTelemetry
```

这样 scheduler 才能决定每个 job 应该跑在 CPU、NPU、dNPU，还是 memory/flash pipeline 上，而不是把整次请求固定给一个 backend。

## 平台策略

### Raspberry Pi / ARM CPU Fallback

Raspberry Pi 是早期没有 Rockchip 或 Qualcomm 板子时的验证平台。它用于验证离线运行结构、GGUF 模型路径、本地 API、benchmark trace、manifest 加载和 CPU fallback 行为（ARM/NEON 正确性、内存、存储 mmap）。

它不验证生产级 NPU 调度，也不拿相对 llama.cpp 的 tok/s 当验收。CPU 对照与差距标准见 `README.plan.zh.md`「CPU fallback 验收标准」。

### Rockchip RK3588 / RK3576

未来主要 backend 候选：

- RKLLM
- RKNN

预期部署路径：

1. 准备 Hugging Face checkpoint。
2. 使用 RKLLM-Toolkit 转换和量化。
3. 生成 `.rkllm` 或 RKNN-compatible artifact。
4. 通过 EdgeXPU backend adapter 加载 artifact。
5. 向 scheduler 汇报 NPU 兼容性、内存占用、延迟和 fallback 原因。

### Qualcomm Snapdragon / Dragonwing

未来主要 backend 候选：

- Qualcomm QNN
- ExecuTorch Qualcomm backend
- ONNX Runtime GenAI with QNN execution provider
- 必要时接入 vendor-specific Genie 或 AI Hub artifact

Qualcomm 路径重点验证 artifact 兼容性、operator 支持，以及 Hexagon NPU 上 prefill/decode 的真实性能表现。

### 其他 NPU / dNPU 候选平台记录

除 Rockchip 和 Qualcomm 之外，后续可以持续观察这些带 NPU、APU、BPU、KPU 或独立 AI accelerator 的平台：

- MediaTek Dimensity / Kompanio：带 APU/NPU，手机、平板和 Chromebook 平台较多，但开放工具链和本地 LLM 部署路径需要进一步验证。
- NXP i.MX 8M Plus / i.MX 9：带 NPU，偏工业边缘设备，适合小模型和稳定部署场景。
- Amlogic A311D2 / Amlogic NPU 系列：部分开发板带 NPU，生态和 LLM 工具链成熟度需要评估。
- Kendryte K230 / K510：RISC-V + KPU/NPU，适合小模型、视觉和轻量 AI 工作负载。
- Sophgo BM1684X / CV 系列：偏边缘盒子、算力卡和国产化部署路线，主要工具链是 Sophon SDK。
- Axera AX 系列：国内边缘 AI SoC，偏视觉、多模态和端侧推理。
- Horizon Robotics Journey / Sunrise：车载和机器人方向，使用 BPU 架构。
- Intel Core Ultra NPU：x86 边缘设备上的 NPU 路线，可以通过 OpenVINO 方向评估。
- Google Coral Edge TPU / Hailo-8：更适合轻量模型、embedding、vision encoder 或辅助任务，不应作为主 LLM backend 的第一优先级。

初步优先级：

1. Rockchip RKLLM/RKNN：优先验证，适合当前项目的第一条 NPU 路线。
2. Qualcomm QNN：第二优先级，适合 Snapdragon / Dragonwing 设备和未来移动端路线。
3. NXP / MediaTek / Intel OpenVINO：作为中期候选，重点看工具链开放程度和 transformer operator 支持。
4. Sophgo / Axera / Horizon：作为国产边缘 AI SoC 路线储备，适合后续按硬件可得性推进。
5. Hailo / Coral：主要作为轻量 AI 或视觉/embedding 辅助 accelerator，不作为主 LLM backend。

选型时不能只看是否“带 NPU”。EdgeXPU-LLM 更关心：

- 是否支持 transformer 关键 operator
- 是否支持适合 LLM 的 quantization 格式
- prefill 和 decode 是否都能有效执行
- 是否支持 dynamic shape 或可接受的固定 shape 编排
- KV cache 能否被 runtime 管理或绕过
- vendor runtime 是否开放、稳定、可离线部署
- telemetry 是否足够支撑 scheduler 做决策

## 模型契约

每个模型应该由一个逻辑 manifest 描述，并可以关联多个平台 artifact。

示例目录：

```text
models/qwen2.5-1.5b/
  model.manifest.json
  cpu/qwen2.5-1.5b-q4.gguf
  rockchip/qwen2.5-1.5b-w8a8-rk3588.rkllm
  qualcomm/qwen2.5-1.5b-qnn/
```

manifest 应该描述：

- 模型身份、family、参数规模、context length
- supported tasks
- artifact format、quantization、backend、path
- memory 和 KV cache 需求
- fallback policy
- 预期吞吐和兼容性说明

应用层不应该关心当前选中的 artifact 跑在 CPU、Rockchip NPU、Qualcomm NPU、dNPU，还是临时 bootstrap backend。

## MVP 范围

MVP 要保持小而可验证。

包含：

- native C runtime skeleton
- 本地 CLI
- OpenAI-compatible `/v1/models` 和 `/v1/chat/completions`
- streaming 和 non-streaming response 形态
- 模型 manifest 加载
- 设备能力 profiling
- 临时 CPU bootstrap backend
- 带近似 prefill/decode trace 的 benchmark 命令
- 完全离线运行

暂不包含：

- 生产级 NPU 执行
- 完整 native transformer 实现
- 高级异构调度
- embeddings
- tool calling
- 模型转换 pipeline
- distributed serving

## 构建和验证

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
bash scripts/verify_mvp.sh
```

日常只跑这一条，全程 `cpu.native`，不调用 llama.cpp。共享 prompt / n 在 `scripts/verify.locks`；各包 greedy id 在 `examples/models/<pack>/verify.lock`。缺 GGUF 的包会跳过，不把 CI 钉死在某一个文件名上。加模型或架构见 `CONTRIBUTING.md`。

llama.cpp 只是可选对照，不进默认验证：

```bash
bash scripts/align_llama.sh              # greedy id vs llama-cli --no-conversation
bash scripts/bench_cpu.sh                # 公平 CPU tok/s，n=32，隐藏 GPU
./build/edgexpu generate <manifest> "<prompt>" 32   # 纯文本；finish_reason 打在 stderr
./build/edgexpu compare <manifest> <prompt>   # 临时 bootstrap 耗时对比，会 shell-out
./build/edgexpu benchmark <manifest> "<prompt>" 32
```

Phase 3.3 ARM / NEON（不进 `verify_mvp.sh`，无相对 llama 的 tok/s 线）：

```bash
sudo apt install gcc-aarch64-linux-gnu qemu-user-static   # 仅 x86 交叉编译需要
bash scripts/verify_arm.sh                                # 交叉编译 + qemu unit；板上直接跑 greedy 锁点
# EDGEXPU_ARM_FULL=1 bash scripts/verify_arm.sh           # x86 上 qemu 跑参考包 greedy（慢）
# 树莓派 5 本机构建再加：-DEDGEXPU_ARM_DOTPROD=ON
```

调试 native 前向（默认 greedy n=8；锁点用 n=4）：

```bash
./build/edgexpu dump-logits examples/models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf "Hello EdgeXPU" 4
```

本地 API：

```bash
./build/edgexpu serve examples/models/qwen2.5-0.5b/model.manifest.json 8000
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @examples/requests/chat_completion.json
```

Windows 用户可以参考 `docs/windows-mingw-setup.md`。

其它 CLI（`capabilities`、`benchmark`、`trace`、自测）见 `./build/edgexpu` 无参数帮助。

`llama` / `llama-cli` 仅在显式跑 `compare` 或 `align_llama.sh` 时需要。

## 计划与设计记录

详细路线图、设计决策、阶段状态、当前限制和近期执行计划统一记录在：

```text
README.plan.zh.md
```

主 `README.md` 只保留项目定位、架构、平台策略和使用方式，避免后续开发记录混在主介绍文档里。

## 隐私和安全

EdgeXPU-LLM 面向私有离线部署。

基本要求：

- 默认不使用云端推理
- 默认不上传 telemetry
- 模型本地存储
- prompt、document、embedding 和 log 本地保存
- 任何网络访问都需要用户显式控制
- 后续支持 signed manifest 和 encrypted model artifact

## 研究参考

- [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](https://arxiv.org/html/2312.12456)
- [PowerInfer-2: Fast Large Language Model Inference on a Smartphone](https://arxiv.org/html/2406.06282)
- [Fast On-device LLM Inference with NPUs](https://arxiv.org/html/2407.05858v2)
- [ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference](https://arxiv.org/html/2508.16703)
- [Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC](https://arxiv.org/html/2506.24045v2)
- [LLM in a Flash: Efficient Large Language Model Inference with Limited Memory](https://machinelearning.apple.com/research/efficient-large-language)
- [RKNN-LLM / RKLLM](https://github.com/airockchip/rknn-llm)
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)
