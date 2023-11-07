FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04

ARG UID
ARG GID
ARG UNAME

# Create a sudo user
RUN groupadd -g $GID -o $UNAME && useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME && usermod -a -G root $UNAME && echo $UNAME:$UNAME | chpasswd && adduser $UNAME sudo && passwd -d $UNAME

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y \
    wget \
    python3.8 \
    python3.8-distutils \
    ffmpeg \
    libsm6 \
    libxext6

RUN wget https://bootstrap.pypa.io/get-pip.py

RUN python3.8 get-pip.py

WORKDIR /ghost 
RUN chown -R $UID:$GID /ghost
USER $UNAME

COPY . .

RUN pip install --upgrade pip setuptools wheel && pip install -r requirements_docker.txt
