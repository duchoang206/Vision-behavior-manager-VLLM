FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# Install build dependencies for YOLO custom parser and other python deps
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    wget \
    python3-pip \
    python3-gi python3-dev python3-gst-1.0 python-gi-dev libglib2.0-dev libgstreamer1.0-dev libgirepository1.0-dev \
    libgstrtspserver-1.0-0 libgstrtspserver-1.0-dev gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-plugins-good \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_x86_64.whl && \
    pip3 install --break-system-packages pyds-1.2.2-cp312-cp312-linux_x86_64.whl && \
    rm pyds-1.2.2-cp312-cp312-linux_x86_64.whl

WORKDIR /app

# Copy backend requirements and install
COPY requirements.txt .
RUN pip3 install --break-system-packages --ignore-installed --no-cache-dir -r requirements.txt

# Copy DeepStream-Yolo source and build the custom parser after dependencies.
# Keeping this after requirements preserves the expensive Python dependency
# layer when only parser/configuration source changes.
COPY models_config/DeepStream-Yolo /app/DeepStream-Yolo
ENV CUDA_VER=12.8
RUN cd /app/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo && \
    make CUDA_VER=${CUDA_VER} && \
    mkdir -p /opt/visionmanager && \
    cp libnvdsinfer_custom_impl_Yolo.so /opt/visionmanager/

# Copy all source files
COPY . /app
RUN g++ -std=c++17 -shared -fPIC -O2 \
    -I/opt/nvidia/deepstream/deepstream/sources/includes \
    -I/usr/local/cuda/include \
    /app/models_config/person_pose_parser.cpp -o /opt/visionmanager/libperson_pose_parser.so

# Ensure pyds (Python bindings) is accessible
ENV PYTHONPATH="/opt/nvidia/deepstream/deepstream/lib:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH}"

# Chỉ cài ultralytics (bản headless để không lỗi libGL) phục vụ việc export sau này
# The requirements layer already installs torch/numpy and their CUDA wheels.
# Avoid --ignore-installed here: it makes pip download a second copy of the
# multi-hundred-megabyte PyTorch/CUDA stack just to install the export helper.
RUN pip3 install --break-system-packages --no-deps --default-timeout=1000 --no-cache-dir ultralytics opencv-python-headless pillow torchvision

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
