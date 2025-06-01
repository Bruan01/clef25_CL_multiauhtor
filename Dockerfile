# 使用官方PyTorch镜像（含CUDA 12.4）
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# 设置工作目录
WORKDIR /app

# 设置环境变量
# ENV HF_ENDPOINT=https://hf-mirror.com
ENV NLTK_DATA=/app/nltk_data
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 配置清华镜像源
RUN pip install --upgrade pip

# 复制并安装Python依赖
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# 创建目录结构
RUN mkdir -p /app/model /app/nltk_data /output /app/deberta_model

# 复制应用代码
COPY submit.py .
COPY model/ /app/model/
COPY nltk_data/ /app/nltk_data/
COPY deberta_model/ /app/deberta_model/

# 设置入口点
ENTRYPOINT ["python", "submit.py"]