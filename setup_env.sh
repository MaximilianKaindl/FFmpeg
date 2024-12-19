#!/bin/bash
# Set root directories for dependencies
export LIBTORCH_ROOT="/home/mkaindl/bsc"
export TOKENIZER_ROOT="/home/mkaindl/bsc"
export FFMPEG_ROOT="/home/mkaindl/bsc"

# Set library paths for runtime
export LD_LIBRARY_PATH="$LIBTORCH_ROOT/libtorch/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$TOKENIZER_ROOT/libtokenizers/lib:$LD_LIBRARY_PATH"


# Configure FFmpeg with all necessary flags
./configure \
    --enable-debug \
    --enable-libtorch \
    --enable-libtokenizers \
    --extra-cflags="-I$LIBTORCH_ROOT/libtorch/include \
                    -I$LIBTORCH_ROOT/libtorch/include/torch/csrc/api/include \
                    -I$TOKENIZER_ROOT/libtokenizers/include" \
    --extra-ldflags="-L$LIBTORCH_ROOT/libtorch/lib \
                     -L$TOKENIZER_ROOT/libtokenizers/lib"   

# Uncomment to build after configuration
# make -j$(nproc)