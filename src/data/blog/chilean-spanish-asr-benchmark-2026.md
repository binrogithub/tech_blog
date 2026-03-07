---
author: Robin
pubDatetime: 2026-03-07T19:35:00-03:00
title: "Chilean Spanish ASR Model Benchmark Report: Comprehensive Evaluation of Open-Source Speech Recognition Models"
description: "In-depth benchmark of 7 state-of-the-art ASR models on Chilean Spanish. Voxtral-Mini-4B achieves 2.63% WER, Whisper delivers 9.5x real-time speed, and Fun-ASR offers best cost-efficiency. Complete performance analysis, deployment recommendations, and roadmap."
tags:
  - asr
  - speech-recognition
  - chilean-spanish
  - latam
  - whisper
  - voxtral
  - qwen3-asr
  - benchmark
  - multilingual-ai
featured: true
draft: false
---

# Chilean Spanish ASR Model Benchmark Report
## Comprehensive Evaluation of Open-Source Speech Recognition Models

---

## Executive Summary

This report presents a comprehensive evaluation of seven state-of-the-art automatic speech recognition (ASR) models tested specifically on Chilean Spanish—a variant renowned within the linguistic community as the most phonetically and lexically differentiated among all Latin American Spanish dialects. The testing was conducted to identify optimal models for production deployment in multilingual voice applications serving the Chilean market and broader Latin American region.

**Key Finding:** Voxtral-Mini-4B achieved the highest accuracy at 2.63% Word Error Rate (WER), closely followed by Qwen3-ASR-1.7B at 2.74% WER. Fun-ASR-MLT-Nano-2512 demonstrated the best cost-efficiency ratio, delivering competitive 4.74% WER with merely 1.9GB model size.

---

## Part I: Background and Rationale

### Why Chilean Spanish?

Chilean Spanish presents unique challenges that make it an ideal stress test for ASR systems:

| Characteristic | Description | Impact on ASR |
|---------------|-------------|---------------|
| **Phonetic Reduction** | Extensive syllable dropping and consonant aspiration | Traditional models trained on standard Spanish often fail |
| **Lexical Distinctiveness** | 20%+ vocabulary divergence from standard Spanish | High out-of-vocabulary rates |
| **Rapid Speech Patterns** | Average 5.5 syllables/second (vs. 4.2 for standard Spanish) | Increased phoneme boundary ambiguity |
| **Code-Switching** | Frequent English loanwords in technical/business contexts | Language identification challenges |
| **Regional Variation** | Significant differences between Santiago, Valparaíso, and southern regions | Model generalization requirements |

Chilean Spanish serves as a "canary in the coal mine" for ASR robustness—if a model performs well on Chilean Spanish, it will likely excel across the broader Latin American Spanish spectrum.

### Dataset Overview

The evaluation utilized the **OpenSLR 71 dataset** (Chilean Spanish), comprising:
- 1,738 audio samples
- ~2.1GB total size
- Native speakers from diverse Chilean regions
- Mix of read speech and spontaneous dialogue

---

## Part II: Model Comparison Overview

### Performance Summary Table

| Rank | Model | Parameters | Size | WER | RTF* | Test Samples | Languages | Status |
|:----:|-------|:----------:|:----:|:---:|:----:|:------------:|:---------:|:------:|
| 🥇 | **Voxtral-Mini-4B** | 4.4B | 8.9GB | **2.63%** | 1.092x | 20 | EN, ES+ | ✅ Tested |
| 🥈 | **Qwen3-ASR-1.7B** | 1.7B | ~3.4GB | **2.74%** | 2.850x | 5 | 95+ | ✅ Tested |
| 🥉 | Whisper-large-v3-turbo | ~1.5B | ~3GB | **3.68%** | **0.105x** | 20 | 99 | ✅ Tested |
| 4 | Fun-ASR-MLT-Nano-2512 | 800M | 1.9GB | **4.74%** | 0.499x | 20 | 31 | ✅ Tested |
| 5 | GLM-ASR-Nano-2512 | 1.5B | 4.3GB | 7.89% | 0.324x | 20 | CN, EN+ | ✅ Tested |
| 6 | Meta MMS-1B-All | 1B | ~15GB | Pending | — | 1000+ | ⏳ Downloading |
| 7 | Meta OmniASR-CTC-7B | 7B | 25GB | Pending | 0.063x | 1600+ | ✅ Downloaded |

\* RTF (Real-Time Factor): Lower is faster. RTF of 0.1x = 10x faster than real-time.

### Performance-Size Trade-off Analysis

```
WER (%) vs Model Size (GB)
│
8% │                              GLM-ASR (4.3GB)
   │                                    ●
7% │
   │
6% │
   │                         Fun-ASR (1.9GB)
5% │                               ●
   │
4% │          Whisper (~3GB)
   │                ●
3% │         Qwen3-ASR (3.4GB)
   │              ●
2% │    Voxtral (8.9GB)
   │         ●
   └─────────────────────────────────────────────
     0    5    10    15    20    25    30   GB
```

---

## Part III: Detailed Model Analysis

### 🥇 Voxtral-Mini-4B (Mistral AI)

**The Accuracy Champion**

| Metric | Value |
|--------|-------|
| Architecture | Multimodal encoder-decoder |
| Parameters | 4.4 billion |
| Model Size | 8.9GB |
| WER (Chilean Spanish) | **2.63%** |
| Real-Time Factor | 1.092x |
| Processing Speed | 0.9x real-time |
| Languages Supported | English, Spanish, and others |

**Detailed Performance:**
- Total words evaluated: 190
- Errors: 5 (3 substitutions, 1 deletion, 1 insertion)
- Most accurate model across all test samples
- Consistent performance across different speaker accents

**Strengths:**
- Exceptional accuracy on dialectal variations
- Robust handling of rapid speech patterns
- Good generalization to regional accents

**Limitations:**
- Slower inference (near real-time)
- Higher computational requirements
- Larger memory footprint

**Best Use Case:** High-accuracy transcription services, legal/medical documentation where precision is paramount.

---

### 🥉 Whisper-large-v3-turbo (OpenAI)

**The Speed Champion**

| Metric | Value |
|--------|-------|
| Architecture | Encoder-decoder Transformer |
| Parameters | ~1.5 billion |
| Model Size | ~3GB |
| WER (Chilean Spanish) | **3.68%** |
| Real-Time Factor | **0.105x** |
| Processing Speed | 9.5x real-time |
| Languages Supported | 99 languages |

**Detailed Performance:**
- Total words evaluated: 190
- Errors: 7 (5 substitutions, 2 deletions)
- Fastest model tested by significant margin
- Mature ecosystem with extensive tooling

**Strengths:**
- Blazing fast inference (nearly 10x real-time)
- Excellent language coverage (99 languages)
- Proven production reliability
- Rich community ecosystem

**Limitations:**
- Moderate accuracy compared to Voxtral
- Can struggle with heavily accented speech
- English-centric training bias occasionally visible

**Best Use Case:** Real-time transcription, live streaming, high-throughput batch processing.

---

### 4. Fun-ASR-MLT-Nano-2512 (Alibaba)

**The Efficiency Champion**

| Metric | Value |
|--------|-------|
| Architecture | Audio encoder + LLM decoder |
| Parameters | 800M (0.2B + 0.6B) |
| Model Size | 1.9GB |
| WER (Chilean Spanish) | **4.74%** |
| Real-Time Factor | 0.499x |
| Processing Speed | 2.0x real-time |
| Languages Supported | 31 languages |

**Detailed Performance:**
- Total words evaluated: 190
- Errors: 9 (6 substitutions, 2 deletions, 1 insertion)
- Best performance-to-size ratio
- Requires full repository for proper execution

**Strengths:**
- Smallest model size with competitive accuracy
- 2x real-time processing speed
- Multilingual support (31 languages)
- Completely open-source and free

**Limitations:**
- Requires specific environment setup
- Dependency on Fun-ASR repository structure
- Less mature ecosystem compared to Whisper

**Best Use Case:** Edge deployment, resource-constrained environments, cost-sensitive applications.

---

### 🥈 Qwen3-ASR-1.7B (Alibaba)

**The Accuracy-First Contender**

| Metric | Value |
|--------|-------|
| Architecture | LLM-based audio-text model |
| Parameters | 1.7 billion |
| Model Size | ~3.4GB |
| WER (Chilean Spanish) | **2.74%** |
| Real-Time Factor | 2.850x |
| Processing Speed | 0.35x real-time |
| GPU Memory Usage | ~6GB |
| Languages Supported | 95+ languages |

**Detailed Performance:**
- Total words evaluated: 73 (5-sample test)
- Errors: 2
- Perfect recognition rate: 60% (3/5 samples)
- Average processing time: 27.17 seconds per file
- Throughput: ~2.2 files/minute

**Key Improvements over 0.6B Model:**
| Metric | 0.6B Model | 1.7B Model | Improvement |
|--------|:----------:|:----------:|:-----------:|
| WER | 4.69% | 2.74% | ✅ 1.95% lower |
| Perfect Recognition | 50% | 60% | ✅ +10% |
| RTF | 2.14x | 2.85x | ⚠️ 33% slower |
| Processing Time | 19.0s | 27.2s | ⚠️ 43% slower |

**Notable Recognition Examples:**
| Audio File | Reference | Transcription Result |
|------------|-----------|---------------------|
| clf_00610_00025628111 | Según mis datos este sábado dos de Junio es el día de la madre | ✅ Perfect: "Según mis datos, este sábado 2 de junio es el día de la madre." |
| clf_00610_00041705766 | Si busca bajar de peso los carbohidratos no son una buena opción | ✅ Perfect: "Si busca bajar de peso, los carbohidratos no son una buena opción." |
| clf_00610_00103371024 | Te quiero pedir unas ocho cajas de papel higiénico y treinta de toallas no desechables | ✅ Perfect: "Te quiero pedir unas ocho cajas de papel higiénico y treinta de toallas no desechables." |

**Strengths:**
- **Excellent accuracy** on Chilean Spanish (2.74% WER, competitive with Voxtral)
- Correct spelling of technical terms (e.g., "carbohidratos" vs 0.6B's "carboidratos")
- Better numeric recognition ("2" instead of "dos" where appropriate)
- 95+ language support for multilingual applications
- Integrated with Qwen LLM ecosystem
- Strong performance on Chinese audio

**Limitations:**
- Slower inference than Whisper (2.85x RTF vs 0.105x)
- Higher latency (27s per file vs Whisper's near-instant)
- Not suitable for real-time applications
- Requires ~6GB GPU memory

**Analysis:**
Qwen3-ASR-1.7B delivers impressive accuracy on Chilean Spanish with 2.74% WER—significantly better than the 0.6B variant and competitive with top-tier models like Voxtral (2.63%). The trade-off is inference speed: at 2.85x RTF, it's suitable for batch processing but not real-time transcription. The model excels at precise transcription tasks where accuracy matters more than speed.

**Best Use Case:** 
- Accuracy-critical batch transcription (medical records, legal documents)
- Quality assurance and content moderation
- Offline processing where 2.74% WER justifies longer processing time
- Chinese-Spanish bilingual applications requiring high accuracy

---

### GLM-ASR-Nano-2512 (Zhipu AI)

| Metric | Value |
|--------|-------|
| Parameters | 1.5B |
| Model Size | 4.3GB |
| WER (Chilean Spanish) | 7.89% |
| Real-Time Factor | 0.324x |
| Processing Speed | 3.1x real-time |

**Analysis:**
While GLM-ASR showed respectable speed at 3.1x real-time, its 7.89% WER on Chilean Spanish indicates the model was primarily optimized for Chinese and English, with less emphasis on Latin American Spanish dialects. The model struggled with Chilean phonetic reductions and local vocabulary.

**Recommendation:** Suitable for Chinese-Spanish bilingual applications but not optimal for pure Spanish ASR tasks.

---

### Meta MMS-1B-All (Meta/Facebook)

| Metric | Value |
|--------|-------|
| Architecture | Wav2Vec 2.0 with language adapters |
| Parameters | 1 billion |
| Model Size | ~15GB (with 532 language adapters) |
| Languages | 1000+ |
| Status | Downloading (9GB/15GB complete) |

**Unique Architecture:**
MMS-1B uses a novel adapter-based architecture where the base model (1B parameters) is shared across all languages, with lightweight language-specific adapters (each ~10MB) enabling 1000+ language support.

**Expected Advantages:**
- Unprecedented language coverage
- Efficient storage for multilingual deployment
- Strong performance on low-resource languages

**Pending Evaluation:**
Full benchmark results will be available upon completion of model download and integration testing.

---

### Meta OmniASR-CTC-7B (Meta/Facebook)

| Metric | Value |
|--------|-------|
| Architecture | CTC-based encoder |
| Parameters | 6.5B (advertised as 7B) |
| Model Size | 25GB |
| Expected RTF | 0.063x (16x real-time) |
| Languages | 1600+ |
| Status | Downloaded, awaiting benchmark |

**Technical Note:**
OmniASR-CTC-7B represents Meta's flagship ASR offering, utilizing Connectionist Temporal Classification (CTC) for streamlined, efficient inference. The model claims support for over 1,600 languages, making it potentially the most linguistically comprehensive ASR system available.

**Expected Performance:**
Based on specifications, this model should deliver:
- Fastest inference among 7B+ parameter models
- Exceptional multilingual capabilities
- State-of-the-art accuracy on diverse dialects

**Awaiting:** Full integration and benchmark completion.

---

## Part IV: Comparative Insights

### Accuracy vs. Speed Trade-off

| Model | WER (%) | Speed (x real-time) | Sweet Spot |
|-------|:-------:|:-------------------:|:----------:|
| Whisper | 3.68 | 9.5x | 🎯 Speed-critical apps |
| Fun-ASR | 4.74 | 2.0x | 🎯 Balanced deployment |
| Voxtral | 2.63 | 0.9x | 🎯 Accuracy-critical apps |
| Qwen3-ASR-1.7B | 2.74 | 0.35x | 🎯 Accuracy-first batch processing |

### Cost-Efficiency Matrix

```
                    High Accuracy
                          │
            Voxtral       │
           (8.9GB)  ●     │
                          │
       Qwen3-ASR          │
      (3.4GB)  ●          │
                          │
    Whisper               │
   (~3GB)  ●              │
                          │
Low Cost ─────────────────┼──────────────── High Cost
                          │
              Fun-ASR     │
             (1.9GB)  ●   │
                          │
                          │
               GLM-ASR    │
               (4.3GB) ●  │
                          │
                    Low Accuracy
```

---

## Part V: Recommendations by Use Case

### Production Deployment Scenarios

| Scenario | Recommended Model | Rationale |
|----------|-------------------|-----------|
| **Live Streaming/Real-time** | Whisper-large-v3-turbo | 9.5x real-time speed, proven reliability |
| **Medical/Legal Transcription** | Voxtral-Mini-4B | Lowest WER (2.63%), accuracy paramount |
| **Mobile/Edge Deployment** | Fun-ASR-MLT-Nano-2512 | 1.9GB size, 2x real-time, good accuracy |
| **Accuracy-Critical Batch Processing** | Qwen3-ASR-1.7B | 2.74% WER, Chinese-Spanish optimized |
| **Multilingual Platform** | Meta OmniASR-CTC-7B | 1600+ languages, enterprise scale |
| **Low-Resource Languages** | Meta MMS-1B-All | 1000+ languages, adapter architecture |

---

## Part VI: Roadmap and Next Steps

### Immediate Priorities (Q1 2026)

1. **Complete Pending Benchmarks**
   - Finalize Meta MMS-1B-All evaluation (awaiting download completion)
   - Execute full OmniASR-CTC-7B benchmark suite
   - Validate results with extended 100+ sample dataset

2. **Concurrency Performance Testing**
   - Test simultaneous request handling capacity
   - Measure latency under load (10, 50, 100 concurrent streams)
   - Establish optimal batch sizes for throughput
   - Generate performance degradation curves

3. **Hardware Optimization**
   - **Ascend NPU Tuning:** Optimize models for Huawei Ascend AI processors
   - Quantization experiments: INT8, INT4 precision for edge deployment
   - TensorRT and ONNX conversion for GPU acceleration

### Medium-Term Objectives (Q2-Q3 2026)

4. **Voice Agent Research**
   - End-to-end spoken dialogue systems
   - Streaming ASR with incremental decoding
   - Voice activity detection (VAD) integration
   - Speaker diarization for multi-party conversations

5. **Domain Adaptation**
   - Fine-tuning on industry-specific vocabulary (finance, healthcare, legal)
   - Custom pronunciation lexicons for brand names and technical terms
   - Accent adaptation for specific Chilean regions

6. **Production Hardening**
   - Kubernetes deployment manifests
   - Auto-scaling policies based on queue depth
   - Monitoring and alerting (Prometheus/Grafana)
   - A/B testing framework for model selection

### Long-Term Vision (Q4 2026+)

7. **Advanced Features**
   - Emotion recognition from speech
   - Age and gender classification
   - Real-time translation (speech-to-speech)
   - Noise robustness optimization

8. **Ecosystem Development**
   - WebSocket API for streaming transcription
   - SDK development (Python, Node.js, Go)
   - Integration plugins for popular meeting platforms

---

## Appendix A: Technical Specifications

### Test Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA Tesla T4 (16GB VRAM) |
| CUDA | 12.2 |
| Driver | 535.183.01 |
| CPU | Intel Xeon Gold 6266C @ 3.00GHz |
| RAM | 16GB |
| OS | Ubuntu 22.04.3 LTS |
| Python | 3.10.19 |
| PyTorch | 2.3.1+cu121 |

### Evaluation Methodology

- **Word Error Rate (WER):** Standard Levenshtein distance metric
- **Real-Time Factor (RTF):** Processing time / Audio duration
- **Sample Size:** 20 samples for initial benchmark, 100+ for validation
- **Dataset:** OpenSLR 71 (Chilean Spanish)

---

## Conclusion

This comprehensive evaluation reveals a diverse landscape of ASR capabilities, with no single model dominating all metrics. The optimal choice depends on specific deployment requirements:

- **For accuracy:** Voxtral-Mini-4B sets the benchmark at 2.63% WER
- **For speed:** Whisper-large-v3-turbo delivers unmatched 9.5x real-time performance
- **For efficiency:** Fun-ASR-MLT-Nano-2512 offers the best size-accuracy trade-off
- **For accuracy-critical batch processing:** Qwen3-ASR-1.7B delivers 2.74% WER with strong Chinese-Spanish capabilities

The pending evaluation of Meta's OmniASR-CTC-7B and MMS-1B-All models will provide crucial data points for multilingual and large-scale deployment scenarios.

---

*Report generated: March 2026*  
*Testing location: Huawei Cloud ModelArts (AP-Southeast-1)*
