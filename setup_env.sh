#!/bin/bash

# Print help message
show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Configure FFmpeg build environment with various dependencies."
    echo
    echo "Options:"
    echo "  --help       Show this help message"
    echo "  --openvino   Include OpenVINO support"
    echo "  --all        Include all dependencies (OpenVINO, CUDA, and additional codecs)"
    echo "  --print-bashrc  Print lines to add to your .bashrc file"
    echo
    echo "No arguments will configure for basic LibTorch and Tokenizers support only."
}

# Print bashrc variables
print_bashrc() {
    # Get absolute paths
    #TODO set paths to your installation directories
    ABSOLUTE_LIBTORCH_ROOT=$(realpath ${LIBTORCH_ROOT:-$(pwd)/../libtorch})
    ABSOLUTE_TOKENIZER_ROOT=$(realpath ${TOKENIZER_ROOT:-$(pwd)/../tokenizers-cpp})
    ABSOLUTE_OPENVINO_ROOT=$(realpath ${OPENVINO_ROOT:-/opt/intel/openvino})

    echo "# Add these lines to your .bashrc file:"
    echo "export LIBTORCH_ROOT=\"${ABSOLUTE_LIBTORCH_ROOT}\""
    echo "export TOKENIZER_ROOT=\"${ABSOLUTE_TOKENIZER_ROOT}\""
    echo "export OPENVINO_ROOT=\"${ABSOLUTE_OPENVINO_ROOT}\""

    # Add library paths
    echo "export PATH=\"\${TOKENIZER_ROOT}/include:\${LIBTORCH_ROOT}/include:\${PATH}\""
    echo "export LD_LIBRARY_PATH=\"\${LIBTORCH_ROOT}/lib:\${TOKENIZER_ROOT}/example/build/tokenizers:\${LD_LIBRARY_PATH}\""

    # Add OpenVINO paths if needed
    if [[ $USE_OPENVINO -eq 1 || $USE_ALL -eq 1 ]]; then
        echo "# OpenVINO environment"
        echo "export PATH=\"\${OPENVINO_ROOT}/runtime/include:\${PATH}\""
        echo "export LD_LIBRARY_PATH=\"\${OPENVINO_ROOT}/runtime/lib/intel64:\${LD_LIBRARY_PATH}\""
        echo "export PKG_CONFIG_PATH=\"\${OPENVINO_ROOT}/runtime/lib/intel64/pkgconfig:\${PKG_CONFIG_PATH}\""
        echo "# Run this line to setup OpenVINO completely:"
        echo "source \"\${OPENVINO_ROOT}/setupvars.sh\""
    fi
}

# Set up paths based on environment variables or defaults
LIBTORCH_ROOT=${LIBTORCH_ROOT:-$(pwd)/../libtorch}
TOKENIZER_ROOT=${TOKENIZER_ROOT:-$(pwd)/../tokenizers-cpp}
OPENVINO_ROOT=${OPENVINO_ROOT:-/opt/intel/openvino}

# Parse command line arguments
USE_OPENVINO=0
USE_ALL=0

if [ "$#" -gt 1 ]; then
    echo "Error: Too many arguments"
    show_help
    exit 1
fi

if [ "$#" -eq 1 ]; then
    case "$1" in
        --help)
            show_help
            exit 0
            ;;
        --openvino)
            USE_OPENVINO=1
            ;;
        --all)
            USE_ALL=1
            USE_OPENVINO=1  # --all includes OpenVINO
            ;;
        --print-bashrc)
            print_bashrc
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
fi

# Setup path variables
setup_paths() {
    # Headers
    LIBTORCH_HEADER=${LIBTORCH_ROOT}/include
    LIBTORCH_HEADER_CSRC=${LIBTORCH_HEADER}/torch/csrc/api/include
    TOKENIZER_HEADER=${TOKENIZER_ROOT}/include
    OPENVINO_HEADER=${OPENVINO_ROOT}/runtime/include

    # Libraries
    LIBTORCH_LIB=${LIBTORCH_ROOT}/lib
    TOKENIZER_LIB=${TOKENIZER_ROOT}/example/build/tokenizers
    OPENVINO_LIB=${OPENVINO_ROOT}/runtime/lib/intel64

    # Update PATH and LD_LIBRARY_PATH
    export PATH="${TOKENIZER_HEADER}:${LIBTORCH_HEADER}:${LIBTORCH_HEADER_CSRC}:${PATH}"
    export LD_LIBRARY_PATH="${LIBTORCH_LIB}:${TOKENIZER_LIB}:${LD_LIBRARY_PATH}"

    if [[ $USE_OPENVINO -eq 1 || $USE_ALL -eq 1 ]]; then
        # Source OpenVINO setupvars.sh if available
        if [ -f "$OPENVINO_ROOT/setupvars.sh" ]; then
            echo "Sourcing OpenVINO environment from $OPENVINO_ROOT/setupvars.sh"
            source "$OPENVINO_ROOT/setupvars.sh"
        else
            export PATH="${OPENVINO_HEADER}:${PATH}"
            export LD_LIBRARY_PATH="${OPENVINO_LIB}:${LD_LIBRARY_PATH}"
            export PKG_CONFIG_PATH="${OPENVINO_LIB}/pkgconfig:${PKG_CONFIG_PATH}"
        fi
    fi
}

# Verify required directories
verify_directories() {
    # Properly declare associative array
    declare -A required_dirs

    required_dirs["LibTorch Library"]=$LIBTORCH_LIB
    required_dirs["Tokenizers Library"]=$TOKENIZER_LIB
    required_dirs["LibTorch Headers"]=$LIBTORCH_HEADER
    required_dirs["Tokenizers Headers"]=$TOKENIZER_HEADER

    if [[ $USE_OPENVINO -eq 1 || $USE_ALL -eq 1 ]]; then
        required_dirs["OpenVINO Library"]=$OPENVINO_LIB
        required_dirs["OpenVINO Headers"]=$OPENVINO_HEADER
    fi

    echo -e "\nVerifying required directories:"
    for desc in "${!required_dirs[@]}"; do
        dir="${required_dirs[$desc]}"
        if [ ! -d "$dir" ]; then
            echo "Error: $desc directory not found at: $dir"
            echo "Please ensure the directory exists and check your environment variables"
            exit 1
        else
            echo "✓ Found $desc at: $dir"
        fi
    done
}

# Generate configuration flags
generate_config_flags() {
    # Base configuration with LibTorch and Tokenizers
    CONFIG_FLAGS="--enable-debug --enable-libtorch --enable-libtokenizers"
    EXTRA_CFLAGS="-I$LIBTORCH_HEADER -I$LIBTORCH_HEADER_CSRC -I$TOKENIZER_HEADER"
    EXTRA_LDFLAGS="-L$LIBTORCH_LIB -L$TOKENIZER_LIB"

    if [[ $USE_OPENVINO -eq 1 ]]; then
        CONFIG_FLAGS="$CONFIG_FLAGS --enable-libopenvino"
        EXTRA_CFLAGS="$EXTRA_CFLAGS -I$OPENVINO_HEADER"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L$OPENVINO_LIB"
    fi

    if [[ $USE_ALL -eq 1 ]]; then
        CONFIG_FLAGS="$CONFIG_FLAGS \
        --enable-gpl \
        --enable-openssl \
        --enable-libx264 \
        --enable-libx265 \
        --enable-libvpx \
        --enable-libfdk-aac \
        --enable-libmp3lame \
        --enable-libopus \
        --enable-libass \
        --enable-libfreetype \
        --enable-libfontconfig \
        --enable-libxcb \
        --enable-sdl2 \
        --enable-cuda-nvcc \
        --enable-libnpp \
        --enable-nonfree"
    fi

    echo -e "\nPreparing FFmpeg configuration with:"
    echo "Flags: $CONFIG_FLAGS"
    echo "CFLAGS: $EXTRA_CFLAGS"
    echo "LDFLAGS: $EXTRA_LDFLAGS"
}

# Main execution flow
main() {
    setup_paths

    # Print configuration summary
    echo "FFmpeg Configuration Summary:"
    if [[ $USE_ALL -eq 1 ]]; then
        echo "- Mode: Full configuration with all dependencies"
    elif [[ $USE_OPENVINO -eq 1 ]]; then
        echo "- Mode: LibTorch, Tokenizers, and OpenVINO"
    else
        echo "- Mode: Basic (LibTorch and Tokenizers only)"
    fi

    verify_directories
    generate_config_flags

    # Run FFmpeg configure
    echo -e "\nRunning FFmpeg configure..."
    ./configure $CONFIG_FLAGS --extra-cflags="$EXTRA_CFLAGS" --extra-ldflags="$EXTRA_LDFLAGS"

    if [ $? -eq 0 ]; then
        echo -e "\nConfiguration successful! Run 'make clean && make -j$(nproc)' to build"
    else
        echo "Error: FFmpeg configuration failed"
        exit 1
    fi
}

# Execute main function
main
