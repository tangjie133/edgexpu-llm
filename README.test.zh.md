# EdgeXPU-LLM 测试记录

这份文件是测试记录，不是项目介绍。项目定位见 `README.md`，阶段计划见 `README.plan.zh.md`。

| 轮次 | 时间 | 对象 |
| --- | --- | --- |
| 第 1 轮 | 2026-09-03 上午 | Phase 3.0–3.2；记下 API/telemetry 问题 |
| 第 2 轮 | 2026-09-03 ~10:00 | P1 修补回归 + ARM qemu greedy 锁点 |
| 第 3 轮 | 2026-09-03 ~10:15 | 第 2 轮 P2 修补回归 |
| 第 4 轮 | 2026-09-03 ~10:50 | 树莓派 4 实机；发现 generate 依赖 llama（P0） |
| 第 5 轮 | 2026-09-03 ~11:10 | P0 修复后：桌面无 llama PATH + 树莓派产品入口 |
| 第 6 轮 | 2026-09-03 ~11:30 | v0.0.7：native 成功不再探测 llama；NEON Q4_K；qemu 自动 emulated |
| 第 7 轮 | 2026-09-03 ~13:45 | 架构插件 / backend vtable / 资源预算；Qwen3.5 识别但不做 native |
| **第 8 轮（本次）** | **2026-09-03 ~14:05** | **`e135898` 需改规则：产品路径禁止 llama 后备；0.5B 包删除；4B 归位** |

测试机（桌面）：Linux x86_64，Intel Core Ultra 9 285K。  
测试机（板子）：**Raspberry Pi 4 Model B Rev 1.4**，aarch64，4 核 Cortex-A72，3.7Gi RAM。仓库 `/home/a/Desktop/edgexpu-llm`（`e135898`）。板上 **无** llama-cli。本轮只记录，不改产品代码。SmolLM GGUF 不进 git；板上本无该文件，测试时从桌面拷入 101MB 后才跑通产品入口。

---

## 0. 第 8 轮结论（`e135898` 需改规则）

规则变成：**产品一律 `cpu.native`；llama 只给 `compare` / `align_llama.sh`。** 未实现的架构必须报缺 adapter，不得改走 llama。Qwen2.5-0.5B 参考包已从仓库去掉；native 参考包是 **SmolLM2-135M**。Qwen3.5 仍 `NATIVE=0`（只锁 tokenize），manifest 改为 `cpu.native`，4B GGUF 已放到 `examples/models/qwen3.5-4b/`。

| 项 | 第 7 轮 | 第 8 轮 |
| --- | --- | --- |
| 桌面 `verify_mvp.sh` | 通过（产品包 SmolLM） | **通过**（同上；Qwen3.5 要求 `artifact.backend: cpu.native` + unsupported） |
| Qwen3.5 `generate`（PATH 上有 llama-cli） | **走 llama**，`[Start thinking]` | **失败**，adapter 错误，exit 1（符合新规则） |
| Qwen3.5 无 llama | 报缺 llama-cli | **同一句 adapter 错误**（不再提 llama-cli） |
| 4B 文件位置 | 在 `qwen2.5-0.5b/` 下用 `../` | **已在 `qwen3.5-4b/`** |
| `EDGEXPU_ARM_FULL=1` | 假绿（空 skip） | **真正跑 SmolLM greedy，通过 1 pack** |
| 板上 greedy | 0.5B MATCH | **SmolLM MATCH**（拷入 GGUF 后）；拷入前 skip |
| 板上 `verify_mvp.sh` | 0.5B 产品包 | 无 SmolLM GGUF 时 **红**；拷入后 **通过** |
| 板上 4B 预算 | admitted=0 | **仍 0**（4351 > 3226） |
| `edgexpu compare` Qwen3.5 | 未强调 | **两腿独立**：native `ok=false`（缺 SSM）；llama `ok=true`（exit 0）。`generate` 仍失败，不走 llama |

### 第 7 轮 P1 本轮

1. **ARM FULL 假绿：已关。** `load_pack` 用 python 读 JSON 路径，不再 exec aarch64 `BIN`。x86 `EDGEXPU_ARM_FULL=1` 打出 `== smollm2-135m greedy lock` 并 **passed (1 pack)**。脚本在「有 native GGUF 但锁点数对不上」时会 `die`。
2. **4B 放错目录：已关。** 桌面 / 板均为 `qwen3.5-4b/Qwen3.5-4B-Q8_0.gguf`。
3. **Qwen3.5 当 llama 产品后端：已关（按新规则）。** 有无 llama-cli 都是 SSM adapter 错误。

### 对得上的行为

- scheduler 只返回 `cpu.native`；load 失败立即返回，不 bind llama。
- CI：`NATIVE=0` 包仍要 `cpu.native` artifact + `native_adapter=unsupported` + tokenize 锁点 MATCH `9419,10041,55,6126`。
- 板上 Qwen3.5 generate 报 adapter，不再报缺 llama-cli。
- chat_template 已带空 `<think>` 块；generate 进不去，无法验收 thinking 行为。
- README 写明 135M 不要用 2+2→4 当能力线：桌面/板 generate 仍是半句 + `length`。板上 n=32 decode **30.72** / prefill **50.7** tok/s，mem 111MB，swap 未用。HTTP `/v1/models` 为 `smollm2-135m`。

### P2

1. **板上默认没有 native 参考权重。** 删掉 0.5B 包后，仓库里唯一 `NATIVE=1` 的 GGUF 是 SmolLM，且 gitignore。未拷文件时 `verify_mvp.sh` 红。3.3 实机入口依赖自备 135M 文件。
2. Qwen3.5 SSM 前向仍未实现（已知缺口）。NPU / Windows / soak 未测。executor 仍单线程。

`compare` 已按「llama 只做独立对照」验收：Qwen3.5 native 失败仍跑 llama 腿；SmolLM 两腿都 `ok`。产品 `generate`/`serve` 不依赖 llama。不要把 compare 的 llama 耗时当 native KPI。

### 建议（给开发，测试不改代码）

1. 板上 README/CONTRIBUTING 写清：要跑 `verify_mvp.sh` / generate，必须自备 `SmolLM2-135M-Instruct-Q4_K_M.gguf`。
2. Qwen3.5 的缺口是 `src/arch/qwen35` 的 SSM 前向，不是再接 llama 当产品后端。

---

## 0a. 第 7 轮结论（`4d55ab7` 修改框架，归档）

新框架按 `CONTRIBUTING.md` 拆开了三层：**模型包**（`examples/models/<pack>/` + `verify.lock`）、**架构插件**（`src/arch/*.c`）、**backend vtable**（`cpu.native` 分步 vs llama 一次性 `generate`）。scheduler 按 mmap 权重 + KV + scratch 估预算，超过设备 RAM **85%** 拒绝 native load。Qwen3.5 hybrid **只识别、不跑 cpu.native**。

官方入口：桌面 `verify_mvp.sh` **通过**；ARM qemu unit **通过**；树上莓派 `verify_arm.sh` greedy **通过**（0.5B）。

| 项 | 结果 |
| --- | --- |
| 桌面 `verify_mvp.sh` | **通过**。skip 缺 GGUF 的 0.5B；Qwen3.5 `NATIVE=0` 只锁 tokenize；**产品包变成 SmolLM** |
| `capabilities` | `arch_plugins=["qwen2","llama","qwen35"]`，`cpu_native=true`；板上 `cpu_baseline=false` |
| `scheduler-selftest` | 桌面 / 板 **budget_reject=ok budget_admit=ok** |
| Qwen3.5 `inspect-gguf` | `architecture=qwen35`，`native_adapter=unsupported`，报错含 Attention+SSM |
| Qwen3.5 tokenize `Hello EdgeXPU` | **MATCH** `9419,10041,55,6126`，roundtrip |
| Qwen3.5 `dump-logits` / `native-selftest` | **拒绝**（同上 adapter 错误），未误跑层循环 |
| Qwen3.5 `generate` 无 llama | **失败**：要 `llama-cli`（包声明 `cpu.baseline`，符合设计） |
| Qwen3.5 `generate` 有 llama（桌面 n=1） | **通过** `backend=cpu.baseline`，文本以 `[Start thinking]` 开头（thinking 模型） |
| 桌面 SmolLM n=32 | decode **81.0** / prefill **219.6** tok/s，mem 111MB；2+2 **答不出**（135M） |
| 板上 0.5B generate / HTTP | **`4` / `stop` / `cpu.native`** |
| 板上 4B 预算 | **`budget_admitted=0`** total_mb=4351 limit_mb=3226（3.7Gi×85%） |
| 板上 greedy | **MATCH**（0.5B）；skip SmolLM（无 GGUF） |
| `EDGEXPU_ARM_FULL=1`（x86） | **假绿**：`load_pack` 直接跑 aarch64 `BIN`，inspect-manifest 失败，锁点全 skip，脚本仍打印 passed |

### 框架行为对得上计划的部分

1. 插件登记可见；`_template` 被 CI skip；CMake 未链 `_template.c`。
2. 缺 GGUF 的包 skip，不把桌面 CI 打红（0.5B 权重仍不在桌面包目录）。
3. hybrid 有明确 unsupported，没有 silently 当 qwen2 跑。
4. 4B 在 Pi 上被预算挡住，inspect 4.2G 文件后 available 仍约 3.3Gi，swap 未用。
5. artifact `backend=cpu.native` → native；`cpu.baseline` → llama 后备。

### P1

1. **x86 上 `EDGEXPU_ARM_FULL=1` 锁点被跳过仍报通过。** `scripts/verify.common.sh` 的 `load_pack` 用 `${BIN} inspect-manifest` 解析 GGUF 路径，而 `verify_arm.sh` 的 `BIN` 是交叉编译的 aarch64 文件。未走 `run_bin`（qemu `-L` sysroot）时，`qemu-aarch64-static` 打不开 `/lib/ld-linux-aarch64.so.1`，路径为空，所有 native 包 `skip missing GGUF`。unit 段是绿的，**greedy 段是空跑**。板上本机不受影响。建议：`load_pack` 在交叉场景用 qemu 包一层，或用 JSON 直接读 artifact.path，不要依赖对 aarch64 二进制的裸 exec。

2. **桌面参考包 0.5B GGUF 仍缺失**，4B 权重放在 `examples/models/qwen2.5-0.5b/Qwen3.5-4B-Q8_0.gguf`。`verify_mvp.sh` 因此把 **SmolLM 135M** 当成 generate/HTTP 产品包。CI 绿，但不能再验收 2+2→4。板上 0.5B 还在。建议：0.5B 文件名放回原包；4B 只放在 `qwen3.5-4b/`。

### P2

1. Qwen3.5 manifest 用 `../qwen2.5-0.5b/Qwen3.5-4B-Q8_0.gguf`，包边界不清。
2. manifest `context_length=8192`，GGUF `context_length=262144`。
3. chat_template 仍是 Qwen2 `<|im_start|>`；llama 生成出现 thinking 标记，模板可能不对齐 Qwen3.5。
4. 板上无 llama，Qwen3.5 **不能**作为板上产品入口（包设计如此）。不要和「默认 cpu.native」混为一谈。
5. 板上仍无 SmolLM GGUF。NPU / Windows / soak 未测。executor 仍单线程。

### 第 7 轮建议（给开发，测试不改代码）

1. 修 `verify_arm.sh` FULL 路径的 `load_pack`，避免 skip 当通过。
2. 把 4B GGUF 挪到自己的包目录，桌面补回 0.5B，让 CI 产品包回到 Coder 0.5B。
3. Qwen3.5 在文档里写清：tokenize + llama 后备；**不是** native 参考包。

---

## 0a. 第 6 轮结论（v0.0.7，归档）

对照第 5 轮剩余项：native 成功后 **不再** 调用 llama `backend->load()`，改为 `edgexpu_backend_cpu_baseline_bind()` 后直接返回。`verify_mvp.sh` 已锁「空 PATH generate」。板上空 PATH `generate` 仍出 `4` / `cpu.native`。

同提交还带了两处非 P0：ARM `Q4_K×Q8_K` NEON 点积；qemu-user 靠 `/proc/cpuinfo` `CPU implementer == 0x00` 自动标 `emulated=true`（不必再设 `EDGEXPU_EMULATED`）。

| 项 | 第 5 轮 | 第 6 轮 |
| --- | --- | --- |
| 提交 | `2f50a1c` v0.0.6 | **`feb32ac` v0.0.7** |
| 桌面 `verify_mvp.sh` | 通过 | **通过**（11:22，当时 0.5B GGUF 仍在；含空 PATH generate） |
| native 成功后是否探测 llama | 会 `backend->load()`（失败忽略） | **否**，`bind()` 后 `return 1` |
| 板上空 PATH `generate` 2+2 | 有 llama 探测但失败被忽略 | **通过** `4` / `stop` / `cpu.native` / ptok=21（PATH 仅空目录） |
| 板上 greedy 锁点 | MATCH | **MATCH**（重建后） |
| 板上 `benchmark` n=32 | decode 8.94 / prefill 16.67 | decode **8.61** / prefill **16.09** / mem **385MB**（同量级，NEON Q4_K **未带来可测加速**） |
| 板上 HTTP 2+2 | `"4"` / `stop` | **通过** `"4"` / `stop` |
| qemu 裸跑 `emulated` | `false`（除非脚本注入） | **`true`**（implementer `0x00`） |
| 桌面 Qwen 参考包 | 0.5B Q4_K_M 在位 | **本轮中途被换成正在下载的 `Qwen3.5-4B-Q8_0.gguf`**，manifest 仍指向旧文件名；之后桌面 Qwen 锁点 / `EDGEXPU_ARM_FULL` **无法复跑** |

板上 capabilities：`arch=aarch64` `cpu_count=4` `memory_total_mb=3796` `simd=neon` `emulated=false` `cpu_baseline=false`。n=32 后 available 仍约 3.3Gi，swap 未用。

### 第 6 轮已关闭

1. **native 成功仍探测 llama。** 已关。产品路径不再要求 PATH 上有 `llama-cli`。
2. **裸 qemu 默认 `emulated=false`。** 已关（aarch64 qemu-user）。实机 implementer 非 0 仍为 `false`。

### 第 6 轮环境问题（不是 runtime 回归）

本轮 11:25 起，桌面 `examples/models/qwen2.5-0.5b/` 里的 `qwen2.5-0.5b-instruct-q4_k_m.gguf` 消失，同目录出现体积持续增大的 `Qwen3.5-4B-Q8_0.gguf`（11:31 已约 3.8G）。manifest 仍写 0.5B 文件名与 `cpu.native`。因此：

- 11:22 的 `verify_mvp.sh` 绿是有效结果；**现在再跑会因缺 0.5B 文件失败**。
- `EDGEXPU_ARM_FULL=1` 报 `missing …qwen2.5-0.5b-instruct-q4_k_m.gguf`。ARM qemu **unit** 仍绿；Qwen greedy 锁点改在 **Pi 本机** 复测并通过。
- 未完成的 Qwen3.5 文件 `inspect-gguf` 为 `GGUF tokenizer tokens 读取失败`。计划已写明换 Qwen3 需要新 adapter，**不要**把这个文件当 0.5B 参考包验收。

建议：4B 权重量放到独立目录；把 0.5B GGUF 放回原文件名，否则桌面 CI 会红。板上 0.5B 包完好。

### 第 6 轮仍存在

1. **Pi 上 NEON Q4_K 无明显 tok/s 收益。** greedy 锁点绿，说明数值没漂；decode 8.61 vs 第 5 轮 8.94，属噪声。若目标是加速，需要另测热点（不一定是这条 dot）。
2. **桌面 Qwen 参考包被换走**（见上）。SmolLM GGUF 仍在，空 PATH generate 走 `cpu.native`（2+2 被 n=16 截成半句，属 135M 质量，不是 P0）。
3. 板上仍无 SmolLM GGUF。NPU / Windows / soak 未测。
4. executor / HTTP 仍单线程；0.5B 幻觉 NVIDIA；无 seed / stop / 真 JSON parser。

---

## 0c. 第 5 轮结论（P0 回归，归档）


第 4 轮 P0 **已关闭**。GGUF load 先 `native_load`，native 成功则不再要求 llama。桌面把 PATH 去掉 `/usr/local/bin`（llama-cli 所在）后 `generate` 仍出 `4`。板上无 llama，`generate` / `benchmark` / `serve` 全部可用，`backend=cpu.native`。

| 项 | 第 4 轮（修前） | 第 5 轮（修后） |
| --- | --- | --- |
| 桌面 `verify_mvp.sh` | 通过（有 llama，掩盖 P0） | **通过** |
| 桌面 `PATH=/usr/bin:/bin generate` | 未测 | **通过** `4` / `stop` / `cpu.native` |
| 板上 `generate` 2+2 | 失败（要 llama-cli） | **通过** `4` / `stop` / `cpu.native` / ptok=21 |
| 板上中文 generate n=16 | 失败 | **通过**，句子被 n 截断，`finish_reason=length` |
| 板上 `benchmark` n=32 | 失败 | **通过** decode **8.94 tok/s**，prefill **16.67**，mem **385MB**，`finish=length` |
| 板上 HTTP | 未测 | **通过**：`/v1/models`；2+2 → `"4"`/`stop`；n=1 → `length`；错误 model → 400 |
| 板上 greedy 锁点 | MATCH | **MATCH**（重建后仍绿） |
| 板上 OOM | 锁点未 OOM | n=32 后 available 仍约 3.3Gi，**swap 未用** |

板上 n=32 文本与桌面同方向（含 NVIDIA 幻觉），属 Coder 0.5B，不是回归。tok/s 只作 telemetry，**无相对 llama 达标线**（板上也没有 llama）。

当时剩余：SmolLM GGUF 仍不在板上；NPU / Windows / soak 未测。load 仍会调用一次 llama `backend->load()`（失败被忽略）。**第 6 轮已去掉这次探测。**

---

## 0b. 第 4 轮结论（树莓派实机，修前，归档）

`scripts/verify_arm.sh` 在板上 **本机构建 + greedy 锁点通过**。NEON、mmap、超窗合同、与 x86 同一把 `verify.locks` 都成立。

**产品入口在板上不可用。** `generate` / `benchmark` / 凡走 `edgexpu_runtime_load_model` 的路径，在没有 `llama-cli` 时直接失败。native 前向其实已经能跑（`dump-logits` / `native-selftest` 成功），但 load 先强制走 llama bootstrap。这和文档「默认 `cpu.native`，llama 只作对照」不一致，记为 **P0**，交给开发修改。本轮未改源码。

| 项 | 板上结果 |
| --- | --- |
| 机型 / 内存 | Pi 4B Rev 1.4，3.7Gi，swap 未用 |
| 本机构建 | Release，OpenMP 4.5，`-march=native`，ELF aarch64 |
| `bash scripts/verify_arm.sh` | **通过**（含 Qwen greedy n=4） |
| dump-logits 锁点 | **MATCH** `9707,10349,55,6325` / `271,2,10349,55` |
| capabilities | `arch=aarch64` `cpu_count=4` `memory_total_mb=3796` `simd=neon` `emulated=false` **`cpu_baseline=false`** |
| GGUF 存储 | `/dev/mmcblk0p2` 上的 380M Qwen Q4_K_M，mmap 完成 prefill/decode（锁点路径） |
| 推理中 OOM | **未发生**（锁点时 available 仍约 3.3Gi） |
| `edgexpu generate` | **失败**（见 P0） |
| `edgexpu benchmark` n=32 | **失败**（同上，未得到板上 tok/s） |
| SmolLM 包 | 板上 **无 GGUF**（只有 manifest） |
| HTTP | 因 generate 加载失败，**未测** |

### P0：generate 仍依赖 llama-cli，不走自研 native

板上报错（原文）：

```text
模型加载失败：未找到 powerinfer、llama-cli 或 manifest 指定的本地二进制
```

对照：

| 入口 | 是否调用 `runtime_load_model` | 板上 |
| --- | --- | --- |
| `dump-logits` / `native-selftest` / `inspect-gguf` | 否，直接 `edgexpu_native_load` | 成功 |
| `generate` / `benchmark` / `trace` / `serve` | 是 | 失败 |

代码路径（供修改 agent，测试未改）：`src/runtime.c` `load_model_job_callback` **先** `runtime->backend->load()`（`src/backend.c` `cpu_baseline_load`，要求 PATH 上有 `powerinfer`/`llama`/`llama-cli`/`main`），失败则 **整次 load 返回**，后面的 `edgexpu_native_load` 根本执行不到。

文档承诺：`README.md` 写 llama 仅 `compare` / `align_llama.sh` 需要；manifest `artifact.backend` 已是 `cpu.native`。实机行为与文档相反。

x86 桌面有 llama-cli，此缺陷被掩盖，`verify_mvp.sh` 绿。板上 `capabilities.runtimes.cpu_baseline=false` 才暴露。

建议（给开发，测试不实现）：GGUF 先 `native_load`；native 成功则不要求 llama；llama 仅 native 失败或显式 bootstrap 时再查 PATH。修完后应在无 llama 的 Pi 上复测 `generate` / `benchmark` / `serve`。

### 第 4 轮环境备注（非产品缺陷）

- 板上原先无 `cmake`，测试时安装了 3.31.6 才能本机构建。
- 板上无 SmolLM GGUF，3.3 锁点只覆盖了 Qwen 参考包。

---

## 1. 第 3 轮结论

官方入口全绿。第 2 轮列出的 P2（telemetry 命名、`finish_reason`、`generate` CLI、HTTP 拼包、qemu `emulated`、Coder 包身份、artifact `cpu.native`）**均已对上**。数值锁点未漂。树莓派实机仍未测。

| 项 | 第 3 轮 |
| --- | --- |
| `bash scripts/verify_mvp.sh` | **通过**（含 pack name、`cpu.native` artifact、`finish_reason`、`telemetry_keep_native`） |
| greedy n=4 x86 | **MATCH** |
| `bash scripts/verify_arm.sh` | **通过**（打印 qemu 核数/内存非板级） |
| `EDGEXPU_ARM_FULL=1` | **ARM greedy lock passed** |
| `edgexpu generate` | stdout 纯文本；stderr `finish_reason=stop backend=cpu.native` |
| HTTP 413 / 分片 body | **413**；两包拼 body 后 `2+2`→`4` |
| 树莓派实机 / NPU | 见第 4 轮：ARM 锁点通过；generate 依赖 llama 未过 / NPU 仍未测 |

---

## 2. 第 2 轮 P2 → 第 3 轮

| 当时建议 | 第 3 轮 |
| --- | --- |
| `telemetry_keep_cpu` 易误解 | **已修。** Next Decode Plan：`policy=telemetry_keep_native`，`backend=cpu.native`。`verify_mvp.sh` 已锁。 |
| `benchmark` 无 `finish_reason` | **已修。** n=32 截断为 `"finish_reason": "length"`；`2+2` generate 为 `stop`。 |
| 交叉编译 format-truncation | **源码已改**（`copy_replay_token` 不再 `snprintf`）。本轮 ARM 增量链接未再刷出该警告。 |
| qemu 核数/内存像板级 | **已修（脚本路径）。** `verify_arm.sh` 设 `EDGEXPU_EMULATED=1` → `"emulated": true`，并打印非 Pi 说明。裸跑 qemu **不**设该变量时仍是 `false`（见下）。 |
| 参考包 Coder 身份 | **已标明。** `name: Qwen2.5-Coder-0.5B-Instruct`，`family=qwen2.5-coder`，`model_id` 仍为 `qwen2.5-0.5b`。 |
| HTTP 单次 recv | **已修。** 按 `Content-Length` 循环读；超缓冲 **413** `request body exceeds server buffer`。分两次 send body 仍能完整推理。 |
| 无 `generate` | **已修。** `edgexpu generate <manifest> <prompt> [n]` |
| manifest artifact 写 baseline | **已修。** `artifact.backend: cpu.native`；`fallback_policy` 仍是 `cpu.baseline`（llama 后备）。 |

`verify_mvp.sh` 已覆盖 name / native artifact / finish_reason / telemetry_keep_native，这些不会静默退回。

---

## 3. 第 3 轮实测摘录

### 3.1 CLI

```text
capabilities (宿主机)
  simd=avx2  emulated=false  arch=x86_64

inspect-manifest
  model_id: qwen2.5-0.5b
  name: Qwen2.5-Coder-0.5B-Instruct
  family: qwen2.5-coder
  artifact.backend: cpu.native
  fallback_policy: cpu.baseline

edgexpu generate … "What is 2+2? Answer with just the number." 16
  stdout: 4
  stderr: finish_reason=stop backend=cpu.native prompt_tokens=21 completion_tokens=1

edgexpu generate … 中文一句话 n=32
  边缘设备上的大语言模型推理是通过深度学习技术，将大量数据输入到模型中，然后进行推理和生成。
  finish_reason=stop  prompt_tokens=19 completion_tokens=27
```

### 3.2 ARM / qemu

```text
qemu 无 EDGEXPU_EMULATED   → 第 3 轮为 emulated=false；第 6 轮起为 true（implementer 0x00）
qemu + EDGEXPU_EMULATED=1 → emulated=true, simd=neon, arch=aarch64
verify_arm.sh 走后者，unit 绿；FULL greedy 锁点绿（需 0.5B GGUF）
```

**注意：** 第 3 轮时忘记设 `EDGEXPU_EMULATED=1`，qemu 二进制看起来像一台 24 核 aarch64 工作站。第 6 轮起 `emulated` 会自动为 true；核数/内存仍是宿主机值。

### 3.3 HTTP

| 用例 | 结果 |
| --- | --- |
| `Content-Length: 40000`（缓冲 32KiB） | **413** Payload Too Large |
| JSON body 拆成两包发送 | `"4"` / `finish_reason=stop` |
| `max_tokens=1` | `"You"` / `length` |

### 3.4 性能（本轮不作 KPI）

同机 `cc1` 占用 100%×多核时，n=32 decode 约 **9–20 tok/s**，prefill **24–38**。第 2 轮空闲是 decode **55.7** / prefill **142**。本轮数字只说明测速必须空闲，**不说明 kernel 回退**。`finish_reason=length` 与 `memory_used_mb=385` 与第 2 轮一致。

未重跑 `bench_cpu.sh`（llama 对照）。

---

## 4. 第 3 轮仍存在的问题

1. **裸 qemu 默认 `emulated=false`。** 第 6 轮已关：qemu-user aarch64 自动 `emulated=true`。
2. **executor / HTTP 仍单线程。** MVP 已知限制。
3. **0.5B 不听 system**（第 2 轮 BANANA）。模板已套上。
4. **生成仍幻觉 NVIDIA**（Coder 0.5B）。身份已写进 manifest，不要当 Instruct 通用助手验收。
5. **无 seed / stop / 真 JSON parser。**
6. **NPU、Windows、soak 未测。** 树莓派见第 5–6 轮：产品入口可用；第 6 轮不再探测 llama。

---

## 5. 建议

1. 第 5–6 轮已关 P0。板上 tok/s 仅记录；不要拿它和桌面或 llama 比达标。
2. qemu-user 现在会自动标 `emulated=true`；`verify_arm.sh` 仍会设 `EDGEXPU_EMULATED=1`。
3. 不要把第 3 轮桌面满载 9–20 tok/s 写进计划当回归。
4. 4B 已在 `qwen3.5-4b/`。板上要跑产品入口需自备 SmolLM GGUF。
5. 第 8 轮：`EDGEXPU_ARM_FULL=1` 已真正锁 SmolLM。`compare` 两腿独立；产品路径不用 llama。

---

## 6. 复现

```bash
bash scripts/verify_mvp.sh
bash scripts/verify_arm.sh
EDGEXPU_ARM_FULL=1 bash scripts/verify_arm.sh

./build/edgexpu generate examples/models/qwen2.5-0.5b/model.manifest.json "What is 2+2? Answer with just the number." 16
./build/edgexpu trace examples/models/qwen2.5-0.5b/model.manifest.json "Say hello in one word."
```

抓取：`/tmp/edgexpu-test3/`（不进仓库）。

---

## 附录：第 1–2 轮摘要

- 第 1 轮：MVP 绿，但 HTTP 丢多轮、`finish_reason` 恒 stop、prefill 误标 baseline、`memory_used_mb=6`。
- 第 2 轮：上述 P1 已修；qemu greedy 锁点绿；空闲 decode 55.7 tok/s；P2 见上表，第 3 轮已关。
