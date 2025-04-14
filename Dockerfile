FROM ubuntu:22.04
FROM nvidia/cuda:12.4.0-base-ubuntu22.04
FROM python:3.10-slim

LABEL authors="MulhamShaheen"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y build-essential


# Set environment variables to ensure nvcc is on the PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}

WORKDIR /app

RUN apt-get install -y git

COPY requirements.txt .

RUN git clone https://github.com/deepseek-ai/Janus.git

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install packaging
RUN pip3 install flash-attn --no-build-isolation

#RUN pip install --upgrade pip && \
#    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "python", "app.py" ]