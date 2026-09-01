# EdgeXPU-LLM

EdgeXPU-LLM is a proposed offline, edge-native LLM inference framework for ARM-based devices with heterogeneous accelerators. The current design focuses on Rockchip RK3588/RK3576 and Qualcomm Snapdragon/Dragonwing platforms, with ARM CPU fallback. STM32-class targets are intentionally excluded from this first framework draft.

The goal is not to copy cloud GPU inference stacks onto smaller hardware. The goal is to build a local runtime that understands edge constraints: NPU execution limits, CPU/NPU/GPU heterogeneity, unified memory bandwidth, flash storage, quantization formats, KV cache pressure, and privacy-sensitive workloads.

## Goals

- Run Qwen, Llama, Phi, Gemma, GPT-like open-weight models locally on edge devices.
- Provide one offline framework across Rockchip, Qualcomm, and ARM CPU fallback targets.
- Expose an OpenAI-compatible local API for applications, IDE agents, local RAG, and tool calling.
- Keep prompts, documents, embeddings, logs, and model execution fully local.
- Use stage-aware scheduling for prefill, decode, verification, and agent workloads.
- Support future PowerInfer-like sparse scheduling, neuron-cluster placement, and flash-aware weight management.

## Target Platforms

### Rockchip RK3588 / RK3576

Primary backend:

- RKLLM
- RKNN

Expected model families:

- Qwen2 / Qwen2.5 / Qwen3
- TinyLlama
- Phi
- Gemma
- selected VLMs where vision encoder and LLM runtime are supported

Typical deployment path:

1. Prepare a Hugging Face checkpoint.
2. Convert and quantize with RKLLM-Toolkit.
3. Generate `.rkllm` artifacts using W8A8 or W4A16.
4. Run through RKLLM Runtime on the board.
5. Expose the runtime through the EdgeXPU backend adapter.

### Qualcomm Snapdragon / Dragonwing

Primary backend candidates:

- Qualcomm QNN
- ExecuTorch Qualcomm backend
- ONNX Runtime GenAI with QNN execution provider
- vendor-specific Genie / AI Hub artifacts where appropriate

Expected model families:

- Qwen
- Llama 3.2
- Phi
- Gemma
- other open-weight decoder-only models that can be quantized and compiled for the target runtime

### ARM CPU Fallback

Primary backend:

- llama.cpp

Expected model format:

- GGUF

Purpose:

- baseline inference
- debugging
- fallback when NPU operators are unsupported
- devices without supported NPU runtime
- benchmark reference against NPU paths

## Architecture

```text
Application Layer
  IDE Agent / Local Chat / RAG / Tool Calling
        |
Unified Local API
  OpenAI-compatible API
  generate / streamGenerate / embed / classify / toolCall
        |
EdgeXPU Runtime Core
  Model Manager
  Capability Profiler
  Stage-Aware Scheduler
  KV Cache Manager
  Memory and Flash Manager
  Security Manager
        |
Backend Adapter Layer
  llama.cpp Backend
  RKLLM Backend
  QNN Backend
  ONNX Runtime GenAI Backend
  ExecuTorch Backend
        |
Hardware Layer
  ARM CPU
  Rockchip NPU
  Qualcomm Hexagon NPU
  GPU
  DRAM / unified memory / flash
```

## Core Components

### Unified Local API

The framework exposes a consistent local API regardless of the underlying backend.

Candidate API surface:

```text
loadModel(model_id, options)
unloadModel(model_id)
generate(request)
streamGenerate(request)
embed(input)
classify(input)
toolCall(request)
getCapabilities()
benchmark(profile)
```

The application should not need to know whether a model is running through RKLLM, QNN, ONNX Runtime, ExecuTorch, or llama.cpp.

### Model Manager

The model manager handles platform-specific artifacts under one logical model identity.

Example layout:

```text
models/qwen2.5-1.5b/
  model.manifest.json
  cpu/qwen2.5-1.5b-q4.gguf
  rockchip/qwen2.5-1.5b-w8a8-rk3588.rkllm
  qualcomm/qwen2.5-1.5b-qnn/
```

The model manifest should describe:

- model family
- parameter size
- context length
- quantization format
- backend compatibility
- memory requirement
- KV cache requirement
- supported tasks
- expected throughput range
- fallback policy

### Capability Profiler

The capability profiler detects the actual device environment at startup.

It should capture:

- CPU architecture and core count
- SIMD support
- NPU type and driver/runtime version
- available DRAM
- flash or storage bandwidth
- supported quantization formats
- supported operators
- thermal and power mode
- installed backend runtimes

The profiler output drives backend selection and scheduling policy.

### Stage-Aware Scheduler

LLM inference has different stages with different hardware behavior.

Prefill:

- processes the prompt in parallel
- compute intensive
- usually benefits from NPU/GPU if the backend supports the graph

Decode:

- generates one token at a time
- memory bandwidth sensitive
- may not always benefit from NPU due to dynamic shapes and scheduling overhead

Verification or speculative decoding:

- can use a small draft model and a larger verifier
- useful for latency-sensitive local agents

The scheduler should make decisions by stage instead of using a fixed backend for the whole request.

### KV Cache Manager

KV cache is a major memory consumer on edge devices.

Required capabilities:

- context length enforcement
- prefix cache
- prompt cache
- sliding window
- quantized KV cache
- cache eviction
- per-session memory limits
- RAG prefix reuse

### Memory and Flash Manager

Edge devices may be limited by DRAM and storage bandwidth.

The memory manager should support:

- model residency budget
- weight paging
- flash-aware prefetch
- contiguous block layout
- hot/cold weight partitioning
- I/O and compute overlap

This component prepares the framework for future PowerInfer-like and LLM-in-a-Flash-style optimizations.

### Backend Adapters

Backend adapters isolate platform-specific runtimes.

Rockchip adapter:

- loads `.rkllm` artifacts
- calls RKLLM Runtime
- reports NPU utilization and memory usage
- streams generated tokens

Qualcomm adapter:

- loads QNN / ExecuTorch / ONNX Runtime GenAI artifacts
- handles QNN runtime initialization
- reports accelerator availability and fallback

llama.cpp adapter:

- loads GGUF models
- provides CPU baseline
- supports fallback inference

## Innovation Points

### Edge-Native Runtime

The framework is designed around edge hardware constraints rather than cloud GPU assumptions.

### Stage-Aware Heterogeneous Scheduling

The runtime treats prefill and decode differently, selecting CPU, NPU, GPU, or hybrid execution based on measured behavior.

### Portable Backend Contract

The framework provides one API and model contract while allowing each platform to use its own optimized runtime.

### Agent-Aware Execution

IDE and agent workloads have foreground and background tasks. The scheduler can prioritize latency-sensitive requests while delaying background analysis or indexing.

### Flash-Aware Model Execution

The design treats flash storage as part of the inference system, enabling future support for models that exceed available memory.

### PowerInfer-Like Sparse Extension

The initial framework does not depend on sparse models, but it leaves room for:

- activation profiling
- hot/cold neuron placement
- neuron-cluster scheduling
- sparse FFN
- sparse attention
- segmented cache

## MVP Scope

The first implementation should focus on a practical, verifiable baseline.

Recommended MVP:

- local OpenAI-compatible API
- llama.cpp CPU backend
- one NPU backend: RKLLM or QNN
- model manifest format
- device capability profiler
- prefill/decode benchmark harness
- KV cache limits
- streaming response support
- fully offline operation

Suggested first models:

- Qwen2.5-1.5B-Instruct
- Qwen3-0.6B
- Qwen3-1.7B
- Llama 3.2 1B
- Llama 3.2 3B

## Roadmap

### Phase 1: Baseline Runtime

- Implement local API server.
- Add llama.cpp backend.
- Add model manifest parser.
- Add simple benchmark command.

### Phase 2: NPU Backend

- Add RKLLM or QNN adapter.
- Add runtime capability detection.
- Compare CPU and NPU prefill/decode behavior.

### Phase 3: Scheduler

- Implement stage-aware backend selection.
- Add prompt cache and KV cache policy.
- Add latency and memory telemetry.

### Phase 4: Edge Optimization

- Add flash-aware weight management.
- Add speculative decoding support.
- Add background and foreground agent priorities.

### Phase 5: Research Extensions

- Add activation profiling.
- Add hot/cold weight placement.
- Add neuron-cluster scheduling.
- Explore sparse and MoE models.

## Privacy and Security

The framework is designed for offline private deployment.

Security requirements:

- no cloud inference by default
- no telemetry by default
- local model storage
- local vector database
- local logs
- encrypted model artifacts where possible
- signed model manifests
- secure runtime configuration
- explicit user control for any network access

## Research References

- PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU  
  https://arxiv.org/html/2312.12456

- PowerInfer-2: Fast Large Language Model Inference on a Smartphone  
  https://arxiv.org/html/2406.06282

- Fast On-device LLM Inference with NPUs  
  https://arxiv.org/html/2407.05858v2

- ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference  
  https://arxiv.org/html/2508.16703

- Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC  
  https://arxiv.org/html/2506.24045v2

- LLM in a Flash: Efficient Large Language Model Inference with Limited Memory  
  https://machinelearning.apple.com/research/efficient-large-language

- RKNN-LLM / RKLLM  
  https://github.com/airockchip/rknn-llm

- ONNX Runtime GenAI  
  https://github.com/microsoft/onnxruntime-genai

## Current Positioning

EdgeXPU-LLM should be positioned as:

> An offline, edge-native LLM inference runtime that unifies ARM CPU fallback, Rockchip NPU, and Qualcomm NPU backends through a common local API, model contract, capability profiler, and stage-aware scheduler.

The framework is feasible as a staged engineering project and has meaningful research potential in heterogeneous scheduling, flash-aware execution, sparse inference, and agent-oriented edge workloads.
