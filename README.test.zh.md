# EdgeXPU-LLM 测试记录

这份文件是测试记录，不是项目介绍。项目定位见 `README.md`，阶段计划见 `README.plan.zh.md`。

| 轮次 | 时间 | 对象 |
| --- | --- | --- |
| 第 1 轮 | 2026-09-03 上午 | Phase 3.0–3.2；记下 API/telemetry 问题 |
| 第 2 轮 | 2026-09-03 ~10:00 | P1 修补回归 + ARM qemu greedy 锁点 |
| **第 3 轮（本次）** | **2026-09-03 ~10:15** | **第 2 轮 P2 修补回归** |

测试机：Linux x86_64，Intel Core Ultra 9 285K，125 GiB。第 3 轮测速时同机有大量 `cc1`/`rg`，**tok/s 不作回归结论**。

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
| 树莓派实机 / NPU | **仍未测** |

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
qemu 无 EDGEXPU_EMULATED   → emulated=false（仍报宿主机 24 核 / 125GiB）
qemu + EDGEXPU_EMULATED=1 → emulated=true, simd=neon, arch=aarch64
verify_arm.sh 走后者，unit 绿；FULL greedy 锁点绿
```

**注意：** 忘记设 `EDGEXPU_EMULATED=1` 时，qemu 二进制看起来像一台 24 核 aarch64 工作站。官方脚本不会踩这个坑；手工 qemu 会。

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

1. **裸 qemu 默认 `emulated=false`。** 只有脚本注入环境变量。可考虑在 aarch64 交叉二进制上探测 qemu（`/proc` 或 `AT_EXECFN`），避免手工漏设。
2. **executor / HTTP 仍单线程。** MVP 已知限制。
3. **0.5B 不听 system**（第 2 轮 BANANA）。模板已套上。
4. **生成仍幻觉 NVIDIA**（Coder 0.5B）。身份已写进 manifest，不要当 Instruct 通用助手验收。
5. **无 seed / stop / 真 JSON parser。**
6. **树莓派实机、NPU、Windows、soak 未测。**

---

## 5. 建议

1. 下一件该做的测试是 **实机 Pi**（`verify_arm.sh` 在 aarch64 上会跑 greedy 锁点），不要在这台满载 285K 上追 tok/s。
2. 手工 qemu 请始终 `EDGEXPU_EMULATED=1`。
3. 不要把本轮 9–20 tok/s 写进计划当回归。

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
