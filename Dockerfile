# 使用官方PyTorch镜像（含CUDA 12.4）
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# 设置镜像源

# 复制并安装Python依赖
RUN apt-get update && apt-get install -y python3-pip
COPY requirements.txt .
RUN pip3 install --break-system-packages torch==2.6.0
RUN pip3 install --break-system-packages -r requirements.txt

# just model *.pth file
COPY model/easy.pth /app/model/ 
COPY model/medium.pth /app/model/ 
COPY model/hard.pth /app/model/ 

RUN python3 -c "import nltk; nltk.download('punkt_tab');"

ENV HF_HUB_OFFLINE=1

# 复制应用代码
COPY submit.py .

# 设置入口点
ENTRYPOINT ["python3", "submit.py"]
