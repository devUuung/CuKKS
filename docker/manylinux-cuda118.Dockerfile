FROM quay.io/pypa/manylinux_2_28_x86_64

RUN yum install -y yum-utils && \
    yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \
    yum install -y \
        cuda-toolkit-11-8 \
        gcc-toolset-11-gcc \
        gcc-toolset-11-gcc-c++ \
        cmake3 \
        openssl-devel \
        ninja-build \
        autoconf \
        automake \
        libtool && \
    yum clean all && \
    ln -sf /usr/bin/cmake3 /usr/bin/cmake || true

# Verify gcc-toolset-11 is installed correctly
RUN test -x /opt/rh/gcc-toolset-11/root/usr/bin/gcc && \
    /opt/rh/gcc-toolset-11/root/usr/bin/gcc --version && \
    test -x /opt/rh/gcc-toolset-11/root/usr/bin/g++ && \
    /opt/rh/gcc-toolset-11/root/usr/bin/g++ --version

ENV PATH=/usr/local/cuda-11.8/bin:/opt/rh/gcc-toolset-11/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/opt/rh/gcc-toolset-11/root/usr/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV CC=/opt/rh/gcc-toolset-11/root/usr/bin/gcc
ENV CXX=/opt/rh/gcc-toolset-11/root/usr/bin/g++
ENV CUDAHOSTCXX=/opt/rh/gcc-toolset-11/root/usr/bin/g++
