#!/bin/bash
# Set root directories for dependencies
export LIBTORCH_ROOT="/home/mkaindl/bsc"
export FFMPEG_ROOT="/home/mkaindl/bsc"

# Set library paths for runtime
export LD_LIBRARY_PATH="$LIBTORCH_ROOT/libtorch/lib:$LD_LIBRARY_PATH"

# Configure FFmpeg with all necessary flags
./configure \
    --enable-debug \
    --enable-libtorch \
    --extra-cflags="-I$LIBTORCH_ROOT/libtorch/include \
                    -I$LIBTORCH_ROOT/libtorch/include/torch/csrc/api/include" \
    --extra-ldflags="-L$LIBTORCH_ROOT/libtorch/lib" 

# After configuration, run:
# make clean
# make -j$(nproc)