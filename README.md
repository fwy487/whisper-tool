# 视频字幕翻译工具

基于 PyQt5 的桌面应用程序，支持视频字幕自动生成、实时录音转文字，以及中英双语字幕翻译。

## 功能特性

- 视频字幕生成：自动提取音频并生成 SRT 字幕文件
- 实时录音转文字：支持麦克风录音实时识别
- 中英双语翻译：集成有道翻译 API，支持字幕翻译
- CUDA 加速：自动检测 NVIDIA 显卡，支持 GPU 加速
- VAD 语音过滤：智能过滤静音片段，提升识别准确度

## 系统要求

- Python 3.8 - 3.12（推荐 3.9 或 3.10）
- Windows / Linux / macOS
- 可选：NVIDIA 显卡（支持 CUDA 加速）

## 安装依赖

### 1. 基础依赖安装

pip install PyQt5 webrtcvad pyaudio SpeechRecognition openai-whisper numpy

### 2. PyTorch 安装（CPU 版本）

如果你的电脑没有 NVIDIA 显卡，或不需要 GPU 加速：

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

### 3. PyTorch 安装（CUDA 11.8 版本，推荐）

如果你的电脑有 NVIDIA 显卡 且想使用 GPU 加速：

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

注意：需要提前安装 CUDA Toolkit 11.8 并配置环境变量

### 4. 可选：安装 Faster-Whisper（更快但可能不稳定）

pip install faster-whisper

注意：当前版本默认关闭 FastWhisper，因为部分环境可能出现崩溃（0xC0000005）

### 5. 完整依赖列表（requirements.txt）

你可以直接创建 requirements.txt 文件：

PyQt5>=5.15.0
webrtcvad>=2.0.10
pyaudio>=0.2.11
SpeechRecognition>=3.10.0
openai-whisper>=20230314
numpy<2,>=1.23.0
torch>=2.0.0

然后一键安装：

pip install -r requirements.txt

## 外部程序依赖

本工具需要以下外部程序，请确保它们已安装并添加到系统 PATH：

FFmpeg - 音频提取、视频合成 - https://ffmpeg.org/download.html
Whisper CLI - 语音识别命令行 - pip install openai-whisper 后自动可用

### FFmpeg 安装检查

安装后请验证：

ffmpeg -version

## 运行程序

python whisper.py

## NumPy 版本注意事项

重要：当前 Whisper 与 NumPy 2.0+ 存在兼容性问题，必须安装 NumPy 1.x 版本：

如果已安装 NumPy 2.x，请先卸载
pip uninstall numpy

安装兼容版本
pip install "numpy<2" --force-reinstall

## 配置文件说明

有道翻译 API 密钥已内置（代码中），如需使用自己的密钥，请修改：

self.app_id = "你的APP_ID"
self.secret_key = "你的SECRET_KEY"

申请地址：https://ai.youdao.com/

## 常见问题

### 1. CUDA 不可用，只能使用 CPU

- 检查是否安装了 CUDA 版本的 PyTorch
- 检查 NVIDIA 驱动是否最新
- 检查 nvidia-smi 是否能正常显示

### 2. 麦克风无法识别

- Windows：确保麦克风权限已开启（设置 → 隐私 → 麦克风）
- Linux：可能需要安装 portaudio：sudo apt-get install portaudio19-dev

### 3. 字幕生成很慢

- 在「生成速度」中选择更快的模型（tiny/base）
- 确保成功启用 CUDA 加速

## 项目结构

.
├── whisper.py          # 主程序（单文件）
├── requirements.txt    # 依赖列表
└── README.md           # 本文件

## 许可证

MIT License
