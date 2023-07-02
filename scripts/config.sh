#!/bin/bash

CWD=$(pwd)

# Root directory.
HUFF_PROJECT_PATH=$(dirname ${CWD})

# Scripts directory.
HUFF_SCRIPTS_PATH="${HUFF_PROJECT_PATH}/scripts"

# Binary directory.
HUFF_BIN_DIR="${HUFF_PROJECT_PATH}/bin"

# CMake build files and cache.
HUFF_BUILD_DIR="${HUFF_PROJECT_PATH}/build"
