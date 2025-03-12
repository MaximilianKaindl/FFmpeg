#!/bin/bash

# Required dependencies for different configurations:
#
# Basic (Required for all configurations):
# sudo apt install -y build-essential pkg-config
#
# For --cuda option:
# sudo apt install -y nvidia-cuda-toolkit
#
# For --openvino option:
# follow instruction on https://github.com/MaximilianKaindl/DeepFFMPEGVideoClassification/tree/classify_movie?tab=readme-ov-file#5-install-openvino-optional
#
# For --codecs option:
# sudo apt install -y libx264-dev libx265-dev libfdk-aac-dev
#
# For --openssl option:
# sudo apt install -y libssl-dev
#
# For --draw option:
# sudo apt install -y libfreetype6-dev libfontconfig1-dev libharfbuzz-dev libxcb1-dev libsdl2-dev libass-dev
#
# For additional codec libraries (used with --all):
# sudo apt install -y libvpx-dev libmp3lame-dev libopus-dev libaom-dev


# Print help message
show_help() {
    echo "Usage: $0 [OPTION]..."
    echo "Configure FFmpeg build environment with various dependencies."
    echo
    echo "Options:"
    echo "  --help            Show this help message"
    echo "  --openvino        Include OpenVINO support"
    echo "  --cuda            Use LibTorch with CUDA support"
    echo "  --codecs          Enable codec libraries (x264, x265, fdk-aac)"
    echo "  --openssl         Include OpenSSL support"
    echo "  --draw            Include drawing libraries (freetype, fontconfig, harfbuzz, etc.)"
    echo "  --all             Enable all features (shorthand for all available options)"
    echo "  --print-bashrc    Print lines to add to your .bashrc file"
    echo
    echo "Multiple options can be specified together (e.g., --openvino --cuda)"
    echo "No arguments will configure for basic LibTorch and Tokenizers support only."
}

# Print bashrc variables
print_bashrc() {
    # Get absolute paths
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
    if [[ $USE_OPENVINO -eq 1 ]]; then
        echo "# OpenVINO environment"
        echo "export PATH=\"\${OPENVINO_ROOT}/runtime/include:\${PATH}\""
        echo "export LD_LIBRARY_PATH=\"\${OPENVINO_ROOT}/runtime/lib/intel64:\${LD_LIBRARY_PATH}\""
        echo "export PKG_CONFIG_PATH=\"\${OPENVINO_ROOT}/runtime/lib/intel64/pkgconfig:\${PKG_CONFIG_PATH}\""
        echo "# Run this line to setup OpenVINO completely:"
        echo "source \"\${OPENVINO_ROOT}/setupvars.sh\""
    fi
}

# Set up paths based on environment variables or defaults
#TODO set paths to your installation directories
LIBTORCH_ROOT=${LIBTORCH_ROOT:-$(pwd)/../libtorch}
TOKENIZER_ROOT=${TOKENIZER_ROOT:-$(pwd)/../tokenizers-cpp}
OPENVINO_ROOT=${OPENVINO_ROOT:-/opt/intel/openvino}

# Parse command line arguments
USE_OPENVINO=0
USE_ALL=0
USE_CUDA=0
USE_CODECS=0
USE_OPENSSL=0
USE_DRAW=0

# Allow multiple arguments
if [ "$#" -gt 0 ]; then
    for arg in "$@"; do
        case "$arg" in
            --help)
                show_help
                exit 0
                ;;
            --openvino)
                USE_OPENVINO=1
                ;;
            --cuda)
                USE_CUDA=1
                ;;
            --codecs)
                USE_CODECS=1
                ;;
            --openssl)
                USE_OPENSSL=1
                ;;
            --draw)
                USE_DRAW=1
                ;;
            --all)
                USE_ALL=1
                USE_CODECS=1
                USE_OPENSSL=1
                USE_DRAW=1
                USE_CUDA=1
                USE_OPENVINO=1
                ;;
            --print-bashrc)
                print_bashrc
                exit 0
                ;;
            *)
                echo "Error: Unknown option: $arg"
                show_help
                exit 1
                ;;
        esac
    done
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

    if [[ $USE_OPENVINO -eq 1 ]]; then
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

    if [[ $USE_OPENVINO -eq 1 ]]; then
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
    CONFIG_FLAGS="--enable-debug --enable-libtokenizers"
    EXTRA_CFLAGS="-I$LIBTORCH_HEADER -I$LIBTORCH_HEADER_CSRC -I$TOKENIZER_HEADER"
    EXTRA_LDFLAGS="-L$LIBTORCH_LIB -L$TOKENIZER_LIB"
    
    # Configure LibTorch with or without CUDA
    if [[ $USE_CUDA -eq 1 ]]; then
        CONFIG_FLAGS="$CONFIG_FLAGS --enable-libtorch_cuda"
    else
        CONFIG_FLAGS="$CONFIG_FLAGS --enable-libtorch"
    fi

    if [[ $USE_OPENVINO -eq 1 ]]; then
        CONFIG_FLAGS="$CONFIG_FLAGS --enable-libopenvino"
        EXTRA_CFLAGS="$EXTRA_CFLAGS -I$OPENVINO_HEADER"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L$OPENVINO_LIB"
    fi

    if [[ $USE_CODECS -eq 1 ]]; then
        CONFIG_FLAGS="$CONFIG_FLAGS --enable-gpl"
        CONFIG_FLAGS="$CONFIG_FLAGS --enable-libx264 --enable-libx265"
        CONFIG_FLAGS="$CONFIG_FLAGS --enable-libfdk-aac --enable-nonfree"
        CONFIG_FLAGS="$CONFIG_FLAGS \
        --enable-libvpx \
        --enable-libmp3lame \
        --enable-libopus \
        --enable-libaom"

    fi

    if [[ $USE_OPENSSL -eq 1 ]]; then
        CONFIG_FLAGS="$CONFIG_FLAGS --enable-openssl"
    fi

    if [[ $USE_DRAW -eq 1 ]]; then
        CONFIG_FLAGS="$CONFIG_FLAGS \
        --enable-libass \
        --enable-libfreetype \
        --enable-libfontconfig \
        --enable-libharfbuzz \
        --enable-libxcb \
        --enable-sdl2"
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
        echo "- Mode: Full configuration with all features"
    else
        echo "- Mode: Custom configuration with:"
        [[ $USE_CUDA -eq 1 ]] && echo "  • CUDA support"
        [[ $USE_OPENVINO -eq 1 ]] && echo "  • OpenVINO support"
        [[ $USE_CODECS -eq 1 ]] && echo "  • Codec libraries (x264, x265, fdk-aac)"
        [[ $USE_OPENSSL -eq 1 ]] && echo "  • OpenSSL support"
        [[ $USE_DRAW -eq 1 ]] && echo "  • Drawing libraries"
        if [[ $USE_CUDA -eq 0 && $USE_OPENVINO -eq 0 && $USE_GPL -eq 0 && $USE_OPENSSL -eq 0 && $USE_DRAW -eq 0 ]]; then
            echo "  • Basic (LibTorch and Tokenizers only)"
        fi
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