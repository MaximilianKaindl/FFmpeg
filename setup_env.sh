#!/bin/bash

append_to_path() {
    local dir="$1"
    [[ ":$PATH:" != *":$dir:"* ]] && PATH="$dir:$PATH"
}

append_to_ld_path() {
    local dir="$1"
    [[ ":$LD_LIBRARY_PATH:" != *":$dir:"* ]] && LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
}

LIBTORCH_ROOT=${LIBTORCH_ROOT:-$(pwd)/../libtorch}
TOKENIZER_ROOT=${TOKENIZER_ROOT:-$(pwd)/../tokenizers-cpp}

TOKENIZER_LIB=${TOKENIZER_ROOT}/example/build/tokenizers
TOKENIZER_HEADER=${TOKENIZER_ROOT}/include

LIBTORCH_LIB=${LIBTORCH_ROOT}/lib
LIBTORCH_HEADER=${LIBTORCH_ROOT}/include
LIBTORCH_HEADER_CSRC=${LIBTORCH_HEADER}/torch/csrc/api/include

# Append to PATH if not already present
append_to_path "$TOKENIZER_HEADER"
append_to_path "$LIBTORCH_HEADER"
append_to_path "$LIBTORCH_HEADER_CSRC"

# Append to LD_LIBRARY_PATH if not already present
append_to_ld_path "$LIBTORCH_LIB"
append_to_ld_path "$TOKENIZER_LIB"

# Export the updated paths
export PATH
export LD_LIBRARY_PATH

declare -A required_files=(
    ["LibTorch Library"]="$LIBTORCH_LIB"
    ["Tokenizers Library"]="$TOKENIZER_LIB"
    ["LibTorch Headers"]="$LIBTORCH_HEADER"
    ["Tokenizers Headers"]="$TOKENIZER_HEADER"
)

# Check each required directory with descriptive messages
for desc in "${!required_files[@]}"; do
    dir="${required_files[$desc]}"
    if [ ! -d "$dir" ]; then
        echo "Error: $desc directory not found at: $dir"
        echo "Please ensure the directory exists and check your ROOT path variables"
        exit 1
    else
        echo "✓ Found $desc at: $dir"
    fi
done

# Configure FFmpeg
./configure \
    --enable-debug \
    --enable-libtorch \
    --enable-libtokenizers \
    --extra-cflags="-I$LIBTORCH_HEADER \
                    -I$LIBTORCH_HEADER_CSRC \
                    -I$TOKENIZER_HEADER" \
    --extra-ldflags="-L$LIBTORCH_LIB \
                     -L$TOKENIZER_LIB"

if [ $? -ne 0 ]; then
    echo "Error: FFmpeg configuration failed"
    exit 1
fi

echo "Configuration successful. Run 'make clean && make -j$(nproc)' to build"