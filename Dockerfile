FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
USER root
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY


RUN mkdir -p /root/src

WORKDIR /root/src


RUN apt-get update && apt-get install -y zip unzip git curl tmux valgrind
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs && git lfs install


ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}