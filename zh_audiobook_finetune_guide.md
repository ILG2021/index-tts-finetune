# IndexTTS2 中文有声书微调指南

> **目标配置**：140 小时单人中文有声书 · RTX 5090 · Linux  
> **微调目标**：提升说话风格一致性、降低长文本韵律漂移、改善中文标点停顿

---

## 目录

1. [准备工作](#1-准备工作)
2. [数据集格式要求](#2-数据集格式要求)
3. [第一步：生成 Manifest](#第一步生成-manifest)
4. [第二步：预处理音频，提取特征](#第二步预处理音频提取特征)
5. [第三步：生成 GPT 训练 Pair](#第三步生成-gpt-训练-pair)
6. [第四步：微调训练](#第四步微调训练)
7. [第五步：导出推理 Checkpoint](#第五步导出推理-checkpoint)
8. [第六步：在 WebUI 中使用新模型](#第六步在-webui-中使用新模型)
9. [训练监控](#训练监控)
10. [调参建议](#调参建议)
11. [常见问题](#常见问题)

---

## 1. 准备工作

### 1.1 环境确认

```bash
# 确认 CUDA 可用
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望输出：True NVIDIA GeForce RTX 5090

# 确认 Python 版本 >= 3.10
python --version
```

### 1.2 安装依赖

```bash
# 推荐使用 uv
uv pip install -e .

# 或 pip
pip install -e .
```

### 1.3 预训练权重

确认 `checkpoints/` 目录下存在以下文件：

```
checkpoints/
├── config.yaml
├── gpt.pth            ← 微调起点
├── bpe.model          ← 中文已支持，无需重新训练
├── s2mel.pth
├── wav2vec2bert_stats.pt
└── ...
```

> **重要**：中文 BPE tokenizer 已内置在 `checkpoints/bpe.model` 中，**不需要**重新训练 tokenizer。

---

## 2. 数据集格式要求

### 2.1 音频格式

| 项目 | 推荐值 | 说明 |
|------|--------|------|
| 格式 | WAV / FLAC | 避免 MP3（有损压缩影响特征质量） |
| 采样率 | 22050 Hz 或 16000 Hz | 脚本会自动重采样 |
| 声道 | 单声道 | 双声道会被自动混合 |
| 时长 | 3~30 秒/条 | 过短（<2s）或过长（>60s）会被跳过 |
| 降噪 | 推荐 | 底噪会影响风格学习 |

### 2.2 文本要求

- 纯中文（或中英混合）均可
- 每条对应文本应准确，无漏字错字
- 建议提前处理数字（如 "3" → "三"）

### 2.3 推荐目录结构

```
data/
├── wavs/
│   ├── 001.wav
│   ├── 002.wav
│   └── ...
└── metadata.csv
```

### 2.4 metadata.csv 格式

```csv
audio_path,text,speaker_id,language
wavs/001.wav,今天天气很好，阳光明媚。,speaker_01,zh
wavs/002.wav,她轻轻推开门，走进了那间昏暗的书房。,speaker_01,zh
wavs/003.wav,远处传来一阵悠扬的笛声，令人心旷神怡。,speaker_01,zh
```

**字段说明**：
- `audio_path`：相对于 `--audio-root` 的路径，或绝对路径
- `text`：对应文字
- `speaker_id`：说话人 ID（单人数据填固定值即可）
- `language`：`zh`（中文）

---

## 第一步：生成 Manifest

将 CSV 元数据转换为预处理流水线所需的 JSONL 格式：

```bash
python tools/metadata_to_manifest.py \
  --metadata data/metadata.csv \
  --audio-root data/wavs \
  --output runs/zh_audiobook/manifests/train.jsonl \
  --delimiter "," \
  --default-language zh \
  --default-speaker speaker_01 \
  --skip-missing-audio
```

**验证**：

```bash
head -n 3 runs/zh_audiobook/manifests/train.jsonl
# 期望：{"id":"001","audio":"/abs/path/001.wav","text":"...","language":"zh"}

wc -l runs/zh_audiobook/manifests/train.jsonl
```

---

## 第二步：预处理音频，提取特征

此步骤使用 GPT 模型提取：语义 token、说话人条件嵌入、情感向量、文本 token id。

`--val-count 100` 会从数据集中**固定取 100 条**作为验证集，使用 SHA-1 哈希确保每次断点续传选出的样本一致，**不需要**提前手动切分 manifest。

```bash
python tools/preprocess_multiproc.py \
  --manifest runs/zh_audiobook/manifests/train.jsonl \
  --output-dir runs/zh_audiobook/processed \
  --tokenizer checkpoints/bpe.model \
  --config checkpoints/config.yaml \
  --gpt-checkpoint checkpoints/gpt.pth \
  --language zh \
  --val-count 100 \
  --device cuda \
  --batch-size 8 \
  --workers 4 \
  --num-processes 2 \
  --hf-cache-dir runs/hf_cache
```

**RTX 5090 参数说明**：

| 参数 | 值 | 说明 |
|------|----|------|
| `--val-count` | `100` | 固定 100 条验证集，不受总数比例影响 |
| `--batch-size` | `8` | 5090 32GB VRAM，单次处理 8 条音频 |
| `--workers` | `4` | 每个进程的 DataLoader 线程数 |
| `--num-processes` | `2` | 并行进程数（100 条验证集被均分：每个 worker 负责 50 条）|

> **预计时长**：140 小时数据约需 **2~6 小时**（视 CPU 和磁盘速度）。  
> **断点续传**：中途中断后直接重新运行相同命令，已处理的样本自动跳过，验证集分配结果不变。

**验证预处理结果**：

```bash
ls runs/zh_audiobook/processed/
# 期望：codes/ condition/ emo_vec/ text_ids/ train_manifest.jsonl val_manifest.jsonl

wc -l runs/zh_audiobook/processed/train_manifest.jsonl
wc -l runs/zh_audiobook/processed/val_manifest.jsonl
# 期望：val = 100，train ≈ 总条数 - 100
```

---

## 第三步：生成 GPT 训练 Pair

GPT 的训练范式是：给定一个"提示音频"（prompt），让模型学习生成对应的"目标音频"（target）。

```bash
# 该脚本必须在 tools/ 目录下运行（依赖同目录的 build_gpt_prompt_pairs.py）
cd tools
python generate_gpt_pairs.py \
  --dataset ../runs/zh_audiobook/processed \
  --pairs-per-target 3 \
  --force
cd ..
```

**参数说明**：
- `--pairs-per-target 3`：每条目标音频随机配 3 个提示音频，140 小时数据可生成约 **200万+ 训练对**

**验证**：

```bash
wc -l runs/zh_audiobook/processed/gpt_pairs_train.jsonl
wc -l runs/zh_audiobook/processed/gpt_pairs_val.jsonl
# 期望：val ≈ 300 行（100 条 × 3 pairs），train 数百万行
```

---

## 第四步：微调训练

### 4.1 创建训练脚本

保存为项目根目录的 `train_zh_audiobook.sh`：

```bash
#!/bin/bash
set -e

python trainers/train_gpt_v2.py \
  --train-manifest runs/zh_audiobook/processed/gpt_pairs_train.jsonl::zh \
  --val-manifest   runs/zh_audiobook/processed/gpt_pairs_val.jsonl::zh \
  --tokenizer      checkpoints/bpe.model \
  --config         checkpoints/config.yaml \
  --base-checkpoint checkpoints/gpt.pth \
  --output-dir     runs/zh_audiobook/finetune_ckpts \
  --batch-size     24 \
  --grad-accumulation 2 \
  --epochs         3 \
  --learning-rate  5e-6 \
  --weight-decay   0.01 \
  --warmup-steps   1000 \
  --log-interval   50 \
  --val-interval   2000 \
  --grad-clip      1.0 \
  --text-loss-weight 0.2 \
  --mel-loss-weight  0.8 \
  --amp \
  --resume auto
```

```bash
chmod +x train_zh_audiobook.sh
./train_zh_audiobook.sh
```

### 4.2 RTX 5090 参数详解

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--batch-size` | `24` | 5090 32GB VRAM，单机最大约 24~32 |
| `--grad-accumulation` | `2` | 等效 batch_size=48，平衡显存和效果 |
| `--epochs` | `3` | 140h 数据约 3 轮足够，过多会过拟合 |
| `--learning-rate` | `5e-6` | 微调用小学习率，防止灾难性遗忘 |
| `--warmup-steps` | `1000` | 前 1000 步线性升温 |
| `--amp` | 开启 | 自动混合精度，减少 ~30% 显存 |

> **预计训练时间**：140h 数据 × 3 epoch，batch=48，RTX 5090 约 **20~40 小时**。

### 4.3 显存 OOM 处理

```bash
# 降低 batch size，保持等效 batch=48
--batch-size 16 --grad-accumulation 3
# 或
--batch-size 8 --grad-accumulation 6
```

---

## 第五步：导出推理 Checkpoint

```bash
# 导出最后一个 checkpoint（去掉 optimizer 状态，体积缩小约 3x）
python tools/prune_gpt_checkpoint.py \
  --checkpoint runs/zh_audiobook/finetune_ckpts/latest.pth \
  --output checkpoints/gpt_zh_audiobook.pth
```

**推荐：通过 TensorBoard 找 val/mel_loss 最低的 step 导出**

```bash
# 例如 step 8000 的 val 损失最低
python tools/prune_gpt_checkpoint.py \
  --checkpoint runs/zh_audiobook/finetune_ckpts/model_step8000.pth \
  --output checkpoints/gpt_zh_audiobook_best.pth
```

---

## 第六步：在 WebUI 中使用新模型

### 方法一：临时替换

```bash
mv checkpoints/gpt.pth checkpoints/gpt_base.pth
cp checkpoints/gpt_zh_audiobook.pth checkpoints/gpt.pth
python webui.py
```

### 方法二：并行运行对比

```bash
# 终端 1：原始模型
python webui.py --port 7860 --model_dir checkpoints_base &

# 终端 2：微调模型
python webui.py --port 7861 --model_dir checkpoints_zh
```

---

## 训练监控

```bash
# 安装
pip install tensorboard

# 启动（允许外部访问）
tensorboard --logdir runs/zh_audiobook/finetune_ckpts/logs --host 0.0.0.0 --port 6006
```

访问 `http://<服务器IP>:6006`：

| 指标 | 含义 | 期望趋势 |
|------|------|----------|
| `train/mel_loss` | 语义 token 预测损失 | 持续下降 |
| `train/mel_top1` | Top-1 语义 token 准确率 | 持续上升，目标 >70% |
| `val/mel_loss` | 验证集损失 | 下降后趋于平稳 |
| `train/lr` | 学习率 | warmup 后余弦下降 |

**收敛判断**：`val/mel_loss` 从最低点上升 >5% 即应停止训练，使用最低点的 checkpoint。

---

## 调参建议

### 有声书场景推荐推理参数

| 参数 | 推荐值 | 理由 |
|------|--------|------|
| `temperature` | `0.65` | 微调后模型更稳定，降低随机性 |
| `top_p` | `0.7` | 收窄采样范围，风格更一致 |
| `num_beams` | `3~5` | 有声书场景 5 束更好，但速度变慢 |
| `repetition_penalty` | `12.0` | 减少重复和异常停顿 |
| `max_text_tokens_per_segment` | `80~120` | 有声书短句分割，避免过长 |

### 参考音频选择

微调后对参考音频的依赖降低，但仍建议选择：
- **8~15 秒**的干净片段
- 语速均匀，无明显情绪起伏
- 叙事语调（与有声书风格相近）

---

## 常见问题

### Q1：预处理阶段 CUDA OOM

```bash
--batch-size 4 --workers 2 --num-processes 1
```

### Q2：预处理中断如何续传

直接重新运行相同命令，已处理的样本会被跳过，验证集分配结果因哈希稳定不变。

### Q3：训练 loss 不下降 / NaN

- 学习率过大？尝试 `2e-6`
- 关闭 `--amp` 排查精度问题
- 删除 `processed/` 目录重新预处理

### Q4：微调后效果变差

- 回退到 `val/mel_loss` 反弹前的 checkpoint
- 减少 `--epochs` 到 2

### Q5：generate_gpt_pairs.py 报错找不到模块

```bash
# 必须在 tools/ 目录下运行
cd tools
python generate_gpt_pairs.py \
  --dataset ../runs/zh_audiobook/processed --pairs-per-target 3 --force
cd ..
```

---

## 完整命令速查

```bash
# Step 1: CSV -> Manifest
python tools/metadata_to_manifest.py \
  --metadata data/metadata.csv --audio-root data/wavs \
  --output runs/zh_audiobook/manifests/train.jsonl \
  --delimiter "," --default-language zh --skip-missing-audio

# Step 2: 预处理（RTX 5090 优化，固定 100 条验证集）
python tools/preprocess_multiproc.py \
  --manifest runs/zh_audiobook/manifests/train.jsonl \
  --output-dir runs/zh_audiobook/processed \
  --tokenizer checkpoints/bpe.model --config checkpoints/config.yaml \
  --gpt-checkpoint checkpoints/gpt.pth --language zh \
  --val-count 100 \
  --device cuda --batch-size 8 --workers 4 --num-processes 2 \
  --hf-cache-dir runs/hf_cache

# Step 3: 生成 GPT Pair（在 tools/ 目录执行）
# 自动从 checkpoints/config.yaml 读取 max_mel_tokens / max_text_tokens，
# 过滤超出位置编码表范围的样本，无需手动指定数字。
cd tools
python generate_gpt_pairs.py \
  --dataset ../runs/zh_audiobook/processed --pairs-per-target 3 --force
cd ..

# Step 4: 训练（RTX 5090 优化）
python trainers/train_gpt_v2.py \
  --train-manifest runs/zh_audiobook/processed/gpt_pairs_train.jsonl::zh \
  --val-manifest   runs/zh_audiobook/processed/gpt_pairs_val.jsonl::zh \
  --tokenizer checkpoints/bpe.model --config checkpoints/config.yaml \
  --base-checkpoint checkpoints/gpt.pth \
  --output-dir runs/zh_audiobook/finetune_ckpts \
  --batch-size 24 --grad-accumulation 2 --epochs 3 \
  --learning-rate 5e-6 --weight-decay 0.01 --warmup-steps 1000 \
  --log-interval 50 --val-interval 2000 --grad-clip 1.0 \
  --text-loss-weight 0.2 --mel-loss-weight 0.8 --amp --resume auto

# Step 5: 导出
python tools/prune_gpt_checkpoint.py \
  --checkpoint runs/zh_audiobook/finetune_ckpts/latest.pth \
  --output checkpoints/gpt_zh_audiobook.pth
```

---

*最后更新：2026-04-29 | 适用版本：IndexTTS2 / index-tts-finetune*
