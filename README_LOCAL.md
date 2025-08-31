# ViCrit Local Implementation Guide

This guide helps you run **ViCrit** (Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs) locally using smaller, more manageable models.

## 🎯 What is ViCrit?

ViCrit is a task where Vision-Language Models (VLMs) must identify hallucinations in image descriptions. It's a **verifiable proxy task** for evaluating visual perception capabilities.

**Task**: Given an image + description with one hallucination → Identify the hallucination phrase

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install requirements
pip install -r requirements_local.txt

# Or run the setup script
python setup_local.py
```

### 2. Choose Your Model

**Recommended for most local setups:**
- **LLaVA 7B**: `llava-v1.5-7b` - Good balance of performance and memory usage
- **LLaVA 13B**: `llava-v1.5-13b` - Better performance, needs more VRAM

**Alternative options:**
- **Qwen 7B**: `Qwen/Qwen2.5-VL-7B-Instruct` - Good performance
- **InstructBLIP 7B**: `Salesforce/instructblip-vicuna-7b` - Lightweight option

### 3. Run Evaluation
```bash
# Use the local evaluation script
./evaluation_local.sh

# Or run manually
python model_vicrit_local.py \
    --model_id llava-v1.5-7b \
    --answers-file ./eval_files/local/answers/llava-v1.5-7b.jsonl \
    --batch-size 1 \
    --max-samples 100
```

### 4. Score Results
```bash
python score_local.py --results-file ./eval_files/local/answers/llava-v1.5-7b.jsonl
```

## 📊 Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (for 7B models)
- **RAM**: 16GB system RAM
- **Storage**: 20GB free space

### Recommended Requirements
- **GPU**: 12GB+ VRAM (for 13B models)
- **RAM**: 24GB+ system RAM
- **Storage**: 50GB free space

### Memory Optimization Tips
```bash
# Use 4-bit quantization (if supported)
pip install bitsandbytes

# Use memory-efficient attention
pip install xformers

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## 🔧 Model Configuration

### LLaVA Models
```bash
# 7B model (recommended for most setups)
model_id="llava-v1.5-7b"

# 13B model (better performance)
model_id="llava-v1.5-13b"
```

### Qwen Models
```bash
# 7B model
model_id="Qwen/Qwen2.5-VL-7B-Instruct"

# 14B model (if you have enough VRAM)
model_id="Qwen/Qwen2.5-VL-14B-Instruct"
```

### InstructBLIP Models
```bash
# 7B model (lightweight)
model_id="Salesforce/instructblip-vicuna-7b"

# 13B model
model_id="Salesforce/instructblip-vicuna-13b"
```

## 📁 File Structure

```
ViCrit/
├── model_vicrit_local.py      # Local model implementation
├── evaluation_local.sh        # Local evaluation script
├── score_local.py            # Local scoring script
├── setup_local.py            # Setup helper script
├── requirements_local.txt     # Local requirements
├── eval_files/               # Evaluation outputs
│   └── local/
│       └── answers/
└── README_LOCAL.md           # This file
```

## 🧪 Testing with Limited Samples

Start with a small number of samples to test your setup:

```bash
# Test with only 50 samples
python model_vicrit_local.py \
    --model_id llava-v1.5-7b \
    --answers-file ./test_results.jsonl \
    --max-samples 50
```

## 🔍 Understanding the Output

### Evaluation Results
- **Input**: Image + description with hallucination
- **Output**: Model's prediction of the hallucination phrase
- **Format**: JSONL file with responses and metadata

### Scoring Results
- **Accuracy**: Percentage of correct hallucination identifications
- **Relaxed Correctness**: Partial credit for close matches
- **Detailed Analysis**: Per-sample correctness scores

## 🐛 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```bash
# Reduce batch size
--batch-size 1

# Use smaller model
--model_id llava-v1.5-7b

# Enable memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
```

#### 2. Model Loading Issues
```bash
# Check model name spelling
# Ensure internet connection for model download
# Verify sufficient disk space
```

#### 3. Dataset Download Issues
```bash
# Manual dataset download
python -c "from datasets import load_dataset; load_dataset('russwang/ViCrit-Bench')"
```

### Performance Tips
- **Batch Size**: Start with 1, increase if memory allows
- **Model Selection**: 7B models are usually sufficient for local testing
- **Memory**: Close other applications to free up VRAM

## 📈 Scaling Up

Once your local setup works:

1. **Increase sample count**: Remove `--max-samples` limit
2. **Try larger models**: Move to 13B or 34B if hardware supports
3. **Batch processing**: Increase batch size for faster evaluation
4. **Multiple models**: Compare performance across different architectures

## 🔗 Resources

- **Paper**: [ViCrit: A Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs](https://arxiv.org/abs/2504.07934)
- **Dataset**: [ViCrit-Bench](https://huggingface.co/datasets/russwang/ViCrit-Bench)
- **Training Data**: [ThinkLite-VL-70k](https://huggingface.co/datasets/russwang/ThinkLite-VL-70k)

## 🤝 Contributing

Found issues or have improvements? The local implementation is designed to be:
- **Easy to modify** for different models
- **Memory efficient** for local hardware
- **Well documented** for troubleshooting

## 📝 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{wang2025vicrit,
  title={ViCrit: A Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs},
  author={Wang, Xiyao and Yang, Zhengyuan and Feng, Chao and Liang, Yongyuan and Zhou, Yuhang and Liu, Xiaoyu and Zang, Ziyi and Li, Ming and Lin, Chung-Ching and Lin, Kevin and others},
  journal={arXiv preprint arXiv:2506.10128},
  year={2025}
}
```

---

**Happy evaluating! 🎉** 