# 给贡献者

EdgeXPU-LLM 是边缘离线 LLM runtime。开源后请按层改代码，不要在 `src/native.c` 的层循环里塞新模型。

## 可以改什么

| 你想做的事 | 改这些 | 不要改 |
| --- | --- | --- |
| 换一个已支持架构的 GGUF | `examples/models/<pack>/`（manifest、GGUF、`verify.lock`、chat_template） | runtime / native forward |
| 新 GGUF `general.architecture` | 复制 `src/arch/_template.c` → `src/arch/<id>.c`，实现 `native_forward=1` 的层，在 `register.c` 登记 | 改走 llama `cpu.baseline` 当产品路径 |
| hybrid / SSM / MoE / fused QKV | 同一套 plugin + kernel；未实现前 `generate` 报错 | 把「暂不支持」写成 llama 后备 |
| 新执行后端（RKLLM / QNN） | 新文件实现 `edgexpu_backend` 分步 vtable（`load` / `prefill` / `decode_step`），在 `edgexpu_scheduler_select_backend` 里按 manifest `backend` 选择 | 只在 scheduler 里加字符串、没有 .c |
| tokenizer 不是 GPT-2 BPE | 新 tokenizer 实现 + plugin `configure` 里拒绝或接线 | 假设所有模型都是 gpt2 |

## 模型包

复制 `examples/models/_template/`。

- `model.manifest.json`：`model_id`、artifact `path` / `backend`、`chat_template`
- `verify.lock`：`ADAPTER`、`NATIVE`、`PROMPT_IDS`、可选 `GREEDY_IDS`
- 共享 prompt：`scripts/verify.locks`（`PROMPT` / `GREEDY_N`）
- 缺 GGUF 时 `scripts/verify_mvp.sh` 会 skip 该包，不会红

产品执行**一律** `cpu.native`。llama.cpp 只给 `compare` / `align_llama.sh` 对照，不是某一类模型的正式后端。

`NATIVE=1`：CI 锁 dump-logits（native 已能跑）。`NATIVE=0`：架构插件还未完成前向，CI 只锁 tokenize；`generate` 应失败并指出缺 adapter，**不得**改走 llama。

Qwen3.5 目前是 `NATIVE=0`：能分词，还不能自研 SSM 前向。缺口是实现 `src/arch/qwen35` 的层，而不是声明 `cpu.baseline`。

## 架构插件

1. `src/arch/<id>.c`：`match` + `configure`（RoPE、QKV bias、tensor 名，默认同 llama.cpp `blk.%d.*`）
2. `edgexpu_arch_register_<id>()` 在 `src/arch/register.c` 调用
3. `CMakeLists.txt` 的 `edgexpu_runtime` 加上该 .c（不要提交 `_template.c` 进库）
4. `./build/edgexpu capabilities` 的 `arch_plugins` 应出现你的 id
5. 有 GGUF 时：`inspect-gguf` / `tokenize`；native 可跑再 `dump-logits` 与 `verify.lock`

## Backend 插件

`include/edgexpu/backend.h`：产品 backend 是 `cpu.native` 的分步接口。llama bootstrap 只给 `edgexpu compare` 用。

## 资源调度

`edgexpu_scheduler_estimate_gguf` 估算 mmap 权重 + KV + prefill scratch，超过设备内存 85% 则拒绝 `native_load`。调试可设 `EDGEXPU_BUDGET_DISABLE=1`（不要默认关）。

## 验证

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
bash scripts/verify_mvp.sh
```

不要把 llama.cpp 写进默认 CI。对照数值用 `scripts/align_llama.sh`。
