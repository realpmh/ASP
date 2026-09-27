# ASP — Workshop Evaluator

课堂 Workshop 讨论评估实验系统。核心流程：

1. 课堂双通道录音输入
2. 能量 VAD 分割发言片段
3. MFCC 声纹登记与学生身份匹配
4. 通道时延相关估计发言方向（左 / 中 / 右）
5. 转写 Provider 生成文字
6. 按清晰度、论据、互动性、可执行性形成讨论评分
7. Web 页面展示发言时间线与参与度统计

## 本地运行

```bash
python3 scripts/generate_demo.py
go run ./cmd/server
```

浏览器打开 `http://127.0.0.1:8787`。

## 目录

- `cmd/server`：Go Web/API 服务
- `python/audio_pipeline.py`：音频分析流水线
- `scripts/generate_demo.py`：生成课堂演示音频和学生声纹登记音频
- `data/enroll`：学生声纹登记样本
- `data/demo`：课堂 Workshop 演示音频
- `web`：课程演示界面

## 可替换模型层

课程演示的接口结构预留了以下增强方向：

- ASR：whisper.cpp / faster-whisper / OpenAI-compatible audio API
- 说话人分离：pyannote.audio Community-1
- 说话人识别：SpeechBrain ECAPA-TDNN / WeSpeaker
- 声源定位：GCC-PHAT / 麦克风阵列 DOA
- 评分：本地量表评分器 / OpenAI-compatible LLM API
- 搜索：DuckDuckGo HTML / SearXNG (`SEARXNG_ENDPOINT`)

## 学习助手

右侧“数字音频学习助手”提供轻量搜索接口，适合作为课程实验展示中的论文、开源工具和技术方案查询入口。

## 外部接口配置

```bash
# HTTP ASR
export ASR_API_URL="https://example.com/asr"
export ASR_API_KEY="..."

# whisper.cpp
export WHISPER_CPP_BIN="/path/to/whisper-cli"
export WHISPER_MODEL="/path/to/ggml-model.bin"

# OpenAI-compatible 评分模型
export LLM_API_URL="https://example.com/v1/chat/completions"
export LLM_API_KEY="..."
export LLM_MODEL="your-model"

# 可选 SearXNG 搜索
export SEARXNG_ENDPOINT="https://search.example.com"
```
