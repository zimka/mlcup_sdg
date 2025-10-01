# WorldScore Dockerfile - Evaluation Only
# Optimized for WorldScore evaluation pipeline
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04


# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Accept conda Terms of Service
RUN conda config --set always_yes true && \
    conda config --set auto_activate_base false && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment with specific Python version
RUN conda create -n worldscore python=3.10 -y
ENV CONDA_DEFAULT_ENV=worldscore
ENV CONDA_PREFIX=/opt/conda/envs/worldscore
ENV PATH="${CONDA_PREFIX}/bin:${PATH}"

# Activate conda environment and install PyTorch with CUDA 12.1
RUN conda run -n worldscore conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install additional dependencies
RUN conda run -n worldscore pip install --no-cache-dir \
    torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html \
    --index-url https://download.pytorch.org/whl/cu121 xformers

# Install conda-forge packages
RUN conda run -n worldscore conda install suitesparse -c conda-forge -y

# Install Python packages
RUN conda run -n worldscore pip install --no-cache-dir \
    open3d \
    tensorboard \
    scipy \
    opencv-python \
    tqdm \
    matplotlib \
    pyyaml \
    yacs \
    loguru \
    einops \
    timm \
    imageio \
    spacy \
    catalogue \
    pyiqa \
    torchmetrics \
    pytorch_lightning \
    cvxpy

# Install evo with specific flags
RUN conda run -n worldscore pip install evo --upgrade --no-binary evo

# Install gdown for model downloads
RUN conda run -n worldscore pip install gdown

# Download spacy model
RUN conda run -n worldscore python -m spacy download en_core_web_sm

# Set working directory
WORKDIR /workspace

# Copy the WorldScore repository
COPY WorldScore/ /workspace/

# Create thirdparty directory and clone submodules directly
RUN git submodule update --init --recursive

# Build DROID-SLAM
WORKDIR /workspace/thirdparty/DROID-SLAM
RUN conda run -n worldscore python setup.py install
WORKDIR /workspace

# Build Grounding-SAM with proper environment variables
WORKDIR /workspace/thirdparty/Grounded-Segment-Anything
ENV AM_I_DOCKER=False
ENV BUILD_WITH_CUDA=True
RUN conda run -n worldscore python -m pip install -e segment_anything
RUN conda run -n worldscore pip install --no-build-isolation -e GroundingDINO
WORKDIR /workspace

# Build SAM2
WORKDIR /workspace/thirdparty/sam2
RUN conda run -n worldscore pip install -e .
WORKDIR /workspace

# Install VFIMamba dependencies
RUN conda run -n worldscore pip install causal_conv1d==1.5.0.post8 mamba_ssm==2.2.5

# Install WorldScore package (evaluation only)
RUN conda run -n worldscore pip install -e .

# Create directories for data and models
RUN mkdir -p /workspace/data /workspace/models /workspace/outputs

# Create .env file with default paths
RUN echo "WORLDSCORE_PATH=/workspace" > /workspace/.env && \
    echo "MODEL_PATH=/workspace/models" >> /workspace/.env && \
    echo "DATA_PATH=/workspace/data" >> /workspace/.env

# Download model checkpoints (with retry logic)
RUN conda run -n worldscore mkdir -p /workspace/worldscore/benchmark/metrics/checkpoints

# Download checkpoints with retry logic
RUN for i in {1..3}; do \
    wget -q -P /workspace/worldscore/benchmark/metrics/checkpoints https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth && break || sleep 10; \
    done

RUN for i in {1..3}; do \
    wget -q -P /workspace/worldscore/benchmark/metrics/checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && break || sleep 10; \
    done

RUN for i in {1..3}; do \
    wget -q -P /workspace/worldscore/benchmark/metrics/checkpoints https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip && break || sleep 10; \
    done

RUN unzip /workspace/worldscore/benchmark/metrics/checkpoints/models.zip -d /workspace/worldscore/benchmark/metrics/checkpoints/

RUN for i in {1..3}; do \
    wget -q -P /workspace/worldscore/benchmark/metrics/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt && break || sleep 10; \
    done

RUN for i in {1..3}; do \
    wget -q -P /workspace/worldscore/benchmark/metrics/checkpoints https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt && break || sleep 10; \
    done

RUN for i in {1..3}; do \
    wget -q -P /workspace/worldscore/benchmark/metrics/checkpoints https://huggingface.co/MCG-NJU/VFIMamba_ckpts/resolve/main/ckpt/VFIMamba.pkl && break || sleep 10; \
    done

# Download droid.pth using gdown
RUN conda run -n worldscore gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O /workspace/droid.pth && \
    mv /workspace/droid.pth /workspace/worldscore/benchmark/metrics/checkpoints/

# Set up environment variables
ENV WORLDSCORE_PATH=/workspace
ENV MODEL_PATH=/workspace/models
ENV DATA_PATH=/workspace/data

# Create a simple startup script
#RUN echo '#!/bin/bash' > /workspace/entrypoint.sh && \
#    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /workspace/entrypoint.sh && \
#    echo 'conda activate worldscore' >> /workspace/entrypoint.sh && \
#    echo 'exec "$@"' >> /workspace/entrypoint.sh && \
#    chmod +x /workspace/entrypoint.sh

# Set entrypoint
#ENTRYPOINT ["/workspace/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]