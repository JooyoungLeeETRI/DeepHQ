# Use a base image with Python and CUDA 12.2
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Install essential apt packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    vim \
    tmux \
    htop \
    rsync \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    psmisc \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Copy conda environment yaml file
COPY conda_env.yaml /tmp/conda_env.yaml

# Install and update conda base environment
RUN conda install -y conda

# Update base environment from conda_env.yaml
RUN conda env update -n base --file /tmp/conda_env.yaml && \
    conda clean -afy

# Set default shell to conda environment
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# Install pip packages separately
COPY pip_requirements.txt /tmp/pip_requirements.txt
RUN pip install --no-cache-dir -r /tmp/pip_requirements.txt

# Install additional pip packages excluding fairseq
# RUN grep -v 'fairseq' /tmp/pip_requirements.txt > /tmp/pip_filtered.txt && \
#     pip install --no-cache-dir -r /tmp/pip_filtered.txt

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
