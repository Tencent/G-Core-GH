Build a docker image
===============================

环境主要是 megatron 的运行环境加上 sglang / vllm 的运行环境。从内部摘录出来，略有删减。

The environment mainly consists of the runtime environment for Megatron, along with the runtime environments for sglang and vllm. It has been excerpted from internal sources with some reductions.

## Base Image

首先构建你的基础 docker image，安装 cuda、torch 等。

First, build your base Docker image and install CUDA, Torch, etc.

```bash
readonly space=wepsdl
readonly name=base
readonly base_version='2.7'

function build() {
  CUDA_VERSION_MAJOR=$1
  CUDA_VERSION_MINOR=$2
  CUDNN_VERSION=$3
  TORCH_VERSION=$4
  tag="v${base_version}-cuda-${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}-cudnn-${CUDNN_VERSION}-py-3.10-torch-${TORCH_VERSION}"

  # sometimes you need --no-cache to upgrade cudnn and nccl
  docker build --network=host \
    --progress=plain \
    --build-arg CUDA_VERSION_MAJOR=$CUDA_VERSION_MAJOR \
    --build-arg CUDA_VERSION_MINOR=$CUDA_VERSION_MINOR \
    --build-arg CUDNN_VERSION=$CUDNN_VERSION \
    --build-arg TORCH_VERSION=$TORCH_VERSION \
    -f Dockerfile -t "$space/$name:$tag" .
}
build 12 4 9 2.6.0
```

```dockerfile
# syntax=docker/dockerfile:1

FROM mirrors.tencent.com/tlinux/tlinux3.2:latest

LABEL maintainer="wepsdl"

ARG MY_HOME=/home/wepsdl
USER root
RUN useradd wepsdl

ENV NVARCH="x86_64"
ARG CUDA_VERSION_MAJOR="12"
ARG CUDA_VERSION_MINOR="1"
ENV CUDA_PKG_VERSION="${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}"
ENV PATH="/usr/local/bin:/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/compat:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
ENV CUDA_HOME="/usr/local/cuda"
RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${NVARCH}/7fa2af80.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA && \
  echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict - # buildkit
RUN yum upgrade -y && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel8/$NVARCH/cuda-rhel8.repo > /etc/yum.repos.d/cuda.repo

# install gcc12
RUN yum clean all && \
  yum makecache && \
  dnf install -y scl-utils gcc-toolset-12-gcc gcc-toolset-12-gcc-c++ && \
  echo ". /opt/rh/gcc-toolset-12/enable" >> ~/.bashrc
SHELL [ "/usr/bin/scl", "enable", "gcc-toolset-12"]

# install cuda compat
RUN yum install -y nvidia-container-toolkit && \
  yum install -y cuda-compat-${CUDA_PKG_VERSION} && \
  yum install -y cuda-toolkit-${CUDA_PKG_VERSION}

ARG CUDNN_VERSION="8"
RUN yum install -y libnccl libnccl-devel libcudnn${CUDNN_VERSION}-devel-cuda-${CUDA_VERSION_MAJOR}
 
# install rdma driver
# https://iwiki.woa.com/p/1867675680
# https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
# https://content.mellanox.com/ofed/MLNX_OFED-5.8-2.0.3.0/MLNX_OFED_LINUX-5.8-2.0.3.0-rhel7.2-x86_64.tgz
# https://content.mellanox.com/ofed/MLNX_OFED-5.8-2.0.3.0/MLNX_OFED_LINUX-5.8-2.0.3.0-rhel8.4-x86_64.tgz
RUN yum install -y redhat-rpm-config rpm-build createrepo kernel-rpm-macros libnl3 tk kernel-modules-extra
RUN wget 'https://mirrors.tencent.com/repository/generic/wepsdl/gpu/rdma/MLNX_OFED_LINUX-5.8-2.0.3.0-rhel8.4-x86_64.tgz' -O mlnxofed.tgz \
  && tar -xvf mlnxofed.tgz \
  && cd 'MLNX_OFED_LINUX-5.8-2.0.3.0-rhel8.4-x86_64' \
  && ./mlnxofedinstall --skip-distro-check --distro 'rhel8.4' --user-space-only --force \
  && cd .. \
  && rm -rf mlnxofed.tgz 'MLNX_OFED_LINUX-5.8-2.0.3.0-rhel8.4-x86_64'

# miniconda
RUN wget --no-check-certificate 'https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh' -O /root/miniconda3.sh \
  && bash /root/miniconda3.sh -b -p /root/conda \
  && echo 'export PATH=/root/conda/bin:${PATH}' >>/root/.bashrc \
  && rm /root/miniconda3.sh
ENV PATH=/root/conda/bin:${PATH}
RUN python3 -m pip install --no-cache-dir --upgrade pip

# pytorch
ARG TORCH_VERSION=2.2.2
RUN pip3 install torch==${TORCH_VERSION} torchvision torchaudio numpy==1.26 # --index-url https://download.pytorch.org/whl/cu128
RUN yum install -y ninja-build && pip3 install ninja

CMD ["/bin/bash"]
```

## LLM Image

安装各种 megatron 相关的加速库。我们用 cudnn fused attn 而不是 flash attn 3，为了兼容安装了 flash attn 2。

Install various Megatron-related acceleration libraries. We use cuDNN fused attention instead of Flash Attention 3, but for compatibility, we also install Flash Attention 2.

```bash
readonly space=wepsdl
readonly name=llm
readonly base_version='2.7'
readonly llm_version="${base_version}.11"

function build() {
  CUDA_VERSION_MAJOR=$1
  CUDA_VERSION_MINOR=$2
  CUDNN_VERSION=$3
  TORCH_VERSION=$4
  APEX_VERSION=e13873d
  FLASH_ATTN_VERSION=$5
  TE_VERSION=$6
  tag="v${llm_version}-cuda-${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}-cudnn-${CUDNN_VERSION}-py-3.10-torch-${TORCH_VERSION}-fa-${FLASH_ATTN_VERSION}-te-${TE_VERSION}"

  docker build --network=host \
    --build-arg BASE_VERSION=$base_version \
    --build-arg CUDA_VERSION_MAJOR=$CUDA_VERSION_MAJOR \
    --build-arg CUDA_VERSION_MINOR=$CUDA_VERSION_MINOR \
    --build-arg CUDNN_VERSION=$CUDNN_VERSION \
    --build-arg TORCH_VERSION=$TORCH_VERSION \
    --build-arg APEX_VERSION=$APEX_VERSION \
    --build-arg FLASH_ATTN_VERSION=$FLASH_ATTN_VERSION \
    --build-arg TE_VERSION=$TE_VERSION \
    -f Dockerfile -t "$space/$name:$tag" .
}
build 12 4 9 2.6.0 2.5.9 2.4
```

```dockerfile
# syntax=docker/dockerfile:1
ARG BASE_VERSION="2.3"
ARG CUDA_VERSION_MAJOR="12"
ARG CUDA_VERSION_MINOR="4"
ARG TORCH_VERSION="2.5.1"
ARG CUDNN_VERSION="9"
FROM mirrors.tencent.com/wepsdl/base:v${BASE_VERSION}-cuda-${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}-cudnn-${CUDNN_VERSION}-py-3.10-torch-${TORCH_VERSION}

ARG MY_HOME=/home/wepsdl
WORKDIR ${MY_HOME}
ENV CUDA_HOME="/usr/local/cuda"

ARG BASE_VERSION="2.3"
ARG CUDA_VERSION_MAJOR="12"
ARG CUDA_VERSION_MINOR="4"
ARG TORCH_VERSION="2.5.1"
ARG CUDNN_VERSION="9"

# install apex
ARG APEX_VERSION="24.04.01"
RUN git clone https://mirrors.tencent.com/github.com/NVIDIA/apex && \
  cd apex && \
  git checkout ${APEX_VERSION} && \
  git submodule update --init --recursive && \
  pip3 install -v --disable-pip-version-check --no-build-isolation --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam" ./ && \
  cd .. && \
  rm -rf apex

# instal flash-attn
ARG FLASH_ATTN_VERSION="2.5.9"
RUN git clone https://mirrors.tencent.com/github.com/Dao-AILab/flash-attention.git && \
  cd flash-attention && \
  git checkout v${FLASH_ATTN_VERSION} && \
  git submodule update --init --recursive && \
  MAX_JOBS=50 FLASH_ATTENTION_FORCE_BUILD=TRUE python setup.py install && \
  cd .. && \
  rm -rf flash-attention

# install transformer-engine
ARG TE_VERSION="1.13"
RUN pip3 install git+https://github.com/NVIDIA/nvidia-dlfw-inspect.git@v0.1#egg=nvdlfw-inspect && \
  git clone https://mirrors.tencent.com/github.com/NVIDIA/TransformerEngine.git && \
  cd TransformerEngine && \
  git checkout v${TE_VERSION} && \
  git submodule update --init --recursive && \
  NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 NVTE_CUDA_ARCHS="80;90" NVTE_BUILD_THREADS_PER_JOB=16 python setup.py install && \
  cd .. && \
  rm -rf TransformerEngine

# 0.0.28.post3 for torch 2.5.1
# 0.0.29.post2 for torch 2.6.0
RUN pip3 install xformers==v0.0.29.post2

RUN yum install -y tmux jq dstat git
RUN pip3 install markdown wandb fire tqdm evaluate py-spy jupyter tensorboard matplotlib gradio
RUN pip3 install flask-restful pybind11 zarr==2.18 tensorstore==0.1.45
RUN pip3 install cos-python-sdk-v5>=1.9.25
RUN pip3 install SentencePiece tiktoken lmdb==1.6.2 blobfile==3.0.0 qwen_vl_utils
RUN pip3 install pylatexenc mathruler pynvml
RUN pip3 install datasets==3.0.1 accelerate deepspeed transformers==4.53.2 diffusers==0.34.0
```

## 安装 sglang

```bash
readonly space=wepsdl
readonly name=rl-sglang
readonly llm_version='2.7.11'
readonly rl_sglang_version="${llm_version}.10"

function build() {
  CUDA_VERSION_MAJOR=$1
  CUDA_VERSION_MINOR=$2
  CUDNN_VERSION=$3
  TORCH_VERSION=$4
  APEX_VERSION=e13873d
  FLASH_ATTN_VERSION=$5
  TE_VERSION=$6
  SGLANG_VERSION=$7
  tag="v${rl_sglang_version}-cuda-${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}-cudnn-${CUDNN_VERSION}-py-3.10-torch-${TORCH_VERSION}-fa-${FLASH_ATTN_VERSION}-te-${TE_VERSION}-sglang-${SGLANG_VERSION}"

  docker build --network=host \
    --build-arg LLM_VERSION=$llm_version \
    --build-arg CUDA_VERSION_MAJOR=$CUDA_VERSION_MAJOR \
    --build-arg CUDA_VERSION_MINOR=$CUDA_VERSION_MINOR \
    --build-arg CUDNN_VERSION=$CUDNN_VERSION \
    --build-arg TORCH_VERSION=$TORCH_VERSION \
    --build-arg APEX_VERSION=$APEX_VERSION \
    --build-arg FLASH_ATTN_VERSION=$FLASH_ATTN_VERSION \
    --build-arg TE_VERSION=$TE_VERSION \
    --build-arg SGLANG_VERSION=$SGLANG_VERSION \
    -f Dockerfile -t "$space/$name:$tag" .
}
build 12 4 9 2.6.0 2.5.9 2.4 '0.4.6.post5'
```

```dockerfile
ARG LLM_VERSION="2.3.14"
ARG CUDA_VERSION_MAJOR="12"
ARG CUDA_VERSION_MINOR="4"
ARG CUDNN_VERSION="9"
ARG TORCH_VERSION="2.5.1"
ARG FLASH_ATTN_VERSION='2.5.9'
ARG TE_VERSION='1.13'
FROM mirrors.tencent.com/wepsdl/llm:v${LLM_VERSION}-cuda-${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}-cudnn-${CUDNN_VERSION}-py-3.10-torch-${TORCH_VERSION}-fa-${FLASH_ATTN_VERSION}-te-${TE_VERSION}

ARG SGLANG_VERSION="0.4.6.post3"

RUN pip3 install "sglang[all]==$SGLANG_VERSION" "torch_memory_saver"
```

vllm 和 sglang 安装没有什么大区别，不写了。

There’s not much difference between installing vllm and sglang, so I won’t elaborate on that.
