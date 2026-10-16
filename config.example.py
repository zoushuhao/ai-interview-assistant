# ============================================
# 配置模板（这个文件可以上传 GitHub）
# 使用方法：复制本文件为 config.py，填入你自己的 API Key
#   Windows：copy config.example.py config.py
#   macOS/Linux：cp config.example.py config.py
# config.py 已被 .gitignore 忽略，不会泄露密钥
# ============================================

# ---- 通义千问（当前主力模型）----
# 在 https://bailian.console.aliyun.com 申请
QWEN_API_KEY = "在这里填你的通义千问 API Key"
QWEN_BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"
QWEN_MODEL = "qwen3.8-flash"

# ---- DeepSeek（备用）----
# 在 https://platform.deepseek.com 申请
DEEPSEEK_API_KEY = "在这里填你的 DeepSeek API Key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"
