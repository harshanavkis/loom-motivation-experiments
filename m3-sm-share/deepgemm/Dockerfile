# DeepGEMM next to held SMs (see README.md): CUDA 12.9 (DeepGEMM main asserts CUDA driver/runtime API
# >= 12.9), PyTorch cu129, DeepGEMM built from source. Built on steve:
#   docker build -t loom-deepgemm .
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev git ca-certificates libdw-dev libelf-dev \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu129
RUN pip install --no-cache-dir numpy ninja setuptools wheel
ARG DEEPGEMM_REF=main
RUN git clone --recursive https://github.com/deepseek-ai/DeepGEMM /opt/DeepGEMM && cd /opt/DeepGEMM \
 && git checkout $DEEPGEMM_REF && git submodule update --init --recursive && git rev-parse HEAD > /opt/DeepGEMM.commit \
 && pip install --no-cache-dir --no-build-isolation .
