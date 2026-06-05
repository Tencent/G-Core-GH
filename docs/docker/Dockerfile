# syntax=docker/dockerfile:1

FROM mirrors.tencent.com/tlinux/tlinux3.2:latest

LABEL maintainer="wepsdl"

ARG MY_HOME=/home/wepsdl
USER root
RUN useradd wepsdl

# miniconda
RUN wget --no-check-certificate 'https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh' -O /root/miniconda3.sh \
  && bash /root/miniconda3.sh -b -p /root/conda \
  && echo 'export PATH=/root/conda/bin:${PATH}' >>/root/.bashrc \
  && rm /root/miniconda3.sh
ENV PATH=/root/conda/bin:${PATH}
RUN python3 -m pip install --no-cache-dir --upgrade pip

# https://iwiki.woa.com/p/4008129464
RUN pip config set global.index-url https://mirrors.tencent.com/pypi/simple/ \
  && pip config set global.extra-index-url https://mirrors.tencent.com/tencent_pypi/simple/ \
  && pip config set global.trusted-host mirrors.tencent.com \
  && pip config set global.root-user-action ignore \
  && pip config list

COPY html /root/html

CMD ["/bin/bash"]
