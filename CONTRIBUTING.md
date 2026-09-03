# 给贡献者

EdgeXPU-LLM 是边缘离线 LLM runtime。开源后请按层改代码，不要在 `src/native.c` 的层循环里塞新模型。

## 可以改什么

| 你想做的事 | 改这些 | 不要改 |
| --- | --- | --- |
| 换一个已支持架构的 GGUF | `examples/models/<pack>/`（manifest、GGUF、`verify.lock`、chat_template） | runtime / native forward |
| 新 GGUF `general.architecture` | 复制 `src/arch/_template.c` → `src/arch/<id>.c`，`configure` 填 RoPE / tensor 名，`layer_kind` 标已有 kernel；在 `register.c` 登记 | 在 `src/native.c` 写死该模型的 `blk.%d.*` 或改走 llama |
| hybrid / SSM / MoE / fused QKV | 同一套 plugin + kernel；未实现前 `generate` 报错 | 把「暂不支持」写成 llama 后备 |
| 新执行后端（RKLLM / QNN） | 新文件实现 `edgexpu_backend` 分步 vtable（`load` / `prefill` / `decode_step`），在 `edgexpu_scheduler_select_backend` 里按 manifest `backend` 选择 | 只在 scheduler 里加字符串、没有 .c |
| tokenizer 不是 GPT-2 BPE | 新 tokenizer 实现 + plugin `configure` 里拒绝或接线 | 假设所有模型都是 gpt2 |

## 模型包

复制 `examples/models/_template/`。

- `model.manifest.json`：`model_id`、artifact `path` / `backend`、`chat_template`
- `verify.lock`：`ADAPTER`、`NATIVE`、`PROMPT_IDS`、可选 `GREEDY_IDS`
- 共享 prompt：`scripts/verify.locks`（`PROMPT` / `GREEDY_N`）
- `*.gguf` 不进 git。缺某个包的 GGUF 时 `scripts/verify_mvp.sh` 会 skip **该包**
- 全仓库至少要有一包 `NATIVE=1` 且 GGUF 在位，否则 `verify_mvp.sh` 红（generate/serve 没有产品包）。板上请把 `SmolLM2-135M-Instruct-Q4_K_M.gguf` 放到 `examples/models/smollm2-135m/`

产品执行**一律** `cpu.native`。llama.cpp 只给 `compare` / `align_llama.sh` 对照，不是某一类模型的正式后端。

`NATIVE=1`：CI 锁 dump-logits（native 已能跑）。`NATIVE=0`：架构插件还未完成前向，CI 锁 tokenize，并且 `generate` 必须因缺 adapter 失败。

已有 kernel：`ATTN_SWIGLU`、`GATED_DELTA`、`ATTN_QK_NORM`。新 architecture 只登记 plugin 并填 tensor 名；不要在 `native.c` 加 `load_layer_<model>`。

Qwen3.5 包是 `NATIVE=1` 的 hybrid 示例：plugin 选层类型，forward 走上面三套算子。

## 架构插件

1. `src/arch/<id>.c`：`match` + `configure`（RoPE、QKV bias、tensor 名，默认同 llama.cpp `blk.%d.*`）
2. `edgexpu_arch_register_<id>()` 在 `src/arch/register.c` 调用
3. `CMakeLists.txt` 的 `edgexpu_runtime` 加上该 .c（不要提交 `_template.c` 进库）
4. `./build/edgexpu capabilities` 的 `arch_plugins` 应出现你的 id
5. 有 GGUF 时：`inspect-gguf` / `tokenize`；native 可跑再 `dump-logits` 与 `verify.lock`

## Backend 插件

`include/edgexpu/backend.h`：产品 backend 是 `cpu.native` 的分步接口。`generate` / `serve` / `verify_mvp.sh` 不依赖 llama。

## 资源调度

`edgexpu_scheduler_estimate_gguf` 估算**工作集**（最大 `blk.*` 层 + 输出分块 + embedding 一行 + 注意力层 KV + 有限 scratch），不是整文件 mmap。超过设备内存 85% 则拒绝 `native_load`。调试可设 `EDGEXPU_BUDGET_DISABLE=1`（不要默认关）。verify 在预算拒绝时 skip 该 pack，继续找下一个已准入的 `NATIVE=1` 产品包。

## 验证

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
bash scripts/verify_mvp.sh
```

树莓派 clone 后没有权重。把 `SmolLM2-135M-Instruct-Q4_K_M.gguf` 放到 `examples/models/smollm2-135m/` 再跑 `verify_mvp.sh` 或 `verify_arm.sh`。缺这个文件时脚本红，不是 runtime 回退到 llama。

`verify_mvp.sh` 只验 `cpu.native`。对照脚本 `scripts/align_llama.sh` 是可选的，不进这条入口。
