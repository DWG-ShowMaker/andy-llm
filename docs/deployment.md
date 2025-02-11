# Andy-LLM 部署指南

## 1. Ollama部署

### 1.1 创建Modelfile
```dockerfile
FROM scratch
PARAMETER temperature 0.7
PARAMETER top_k 50
PARAMETER top_p 0.9

# 添加模型文件
ADD best_model.pt /model/
ADD tokenizer.model /model/

# 添加模型配置
TEMPLATE """
{{- if .First }}
[BOS]{{ .Prompt }}
{{- else }}
{{ .Prompt }}
{{- end }}
"""

# 设置系统参数
SYSTEM """
你是一个由Andy-LLM训练的中文语言模型。
"""
```

### 1.2 构建Ollama模型
```bash
# 构建模型
ollama create andy-llm -f Modelfile

# 运行模型
ollama run andy-llm
```