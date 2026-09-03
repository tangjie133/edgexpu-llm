# 给贡献者

EdgeXPU-LLM 是边缘离线 LLM runtime。开源后请按层改代码，不要在 `src/native.c` 的层循环里塞新模型。

## 可以改什么

| 你想做的事 | 改这些 | 不要改 |
| --- | --- | --- |
| 换一个已支持架构的 GGUF | `examples/models/<pack>/`（manifest、GGUF、`verify.lock`、chat_template） | runtime / native forward |
| 新 GGUF `general.architecture`（仍是 dense MHA+SwiGLU） | 复制 `src/arch/_template.c` → `src/arch/<id>.c`，在 `src/arch/register.c` 的 `edgexpu_arch_init` 里登记 | `src/native.c` |
| hybrid / SSM / MoE / fused QKV | 先加 plugin 且 `native_forward=0` 或 `layer_kind=UNSUPPORTED`；实现 kernel 后再打开 | 把 Qwen 特判写进 runtime |
| 新执行后端（RKLLM / QNN） | 新文件实现 `edgexpu_backend` 分步 vtable（`load` / `prefill` / `decode_step`），在 `edgexpu_scheduler_select_backend` 里按 manifest `backend` 选择 | 只在 scheduler 里加字符串、没有 .c |
| tokenizer 不是 GPT-2 BPE | 新 tokenizer 实现 + plugin `configure` 里拒绝或接线 | 假设所有模型都是 gpt2 |

## 模型包

复制 `examples/models/_template/`。

- `model.manifest.json`：`model_id`、artifact `path` / `backend`、`chat_template`
- `verify.lock`：`ADAPTER`、`NATIVE`、`PROMPT_IDS`、可选 `GREEDY_IDS`
- 共享 prompt：`scripts/verify.locks`（`PROMPT` / `GREEDY_N`）
- 缺 GGUF 时 `scripts/verify_mvp.sh` 会 skip 该包，不会红

`NATIVE=1` 要求 `cpu.native` 能 dump-logits。`NATIVE=0` 只锁 tokenize（例如尚未实现的 hybrid）。

## 架构插件

1. `src/arch/<id>.c`：`match` + `configure`（RoPE、QKV bias、tensor 名，默认同 llama.cpp `blk.%d.*`）
2. `edgexpu_arch_register_<id>()` 在 `src/arch/register.c` 调用
3. `CMakeLists.txt` 的 `edgexpu_runtime` 加上该 .c（不要提交 `_template.c` 进库）
4. `./build/edgexpu capabilities` 的 `arch_plugins` 应出现你的 id
5. 有 GGUF 时：`inspect-gguf` / `tokenize`；native 可跑再 `dump-logits` 与 `verify.lock`

## Backend 插件

`include/edgexpu/backend.h`：`engine` 对 `cpu.native` 是 `edgexpu_native_session *`。llama bootstrap 只实现 `load` + `generate`，分步函数为 NULL。

## 资源调度

`edgexpu_scheduler_estimate_gguf` 估算 mmap 权重 + KV + prefill scratch，超过设备内存 85% 则拒绝 `native_load`。调试可设 `EDGEXPU_BUDGET_DISABLE=1`（不要默认关）。

## 验证

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
bash scripts/verify_mvp.sh
```

不要把 llama.cpp 写进默认 CI。对照数值用 `scripts/align_llama.sh`。
