FROM quay.io/pypa/manylinux_2_28_x86_64

RUN yum install -y yum-utils && \
    yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \
    yum install -y \
        cuda-toolkit-12-8 \
        cmake3 \
        openssl-devel \
        ninja-build \
        autoconf \
        automake \
        libtool && \
    yum clean all && \
    ln -sf /usr/bin/cmake3 /usr/bin/cmake || true

ENV PATH=/usr/local/cuda-12.8/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-12.8
