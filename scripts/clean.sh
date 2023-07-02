#!/bin/bash

source config.sh

# Remove the binary directory.
if [ -d $HUFF_BIN_DIR ]
then
    echo "removing '$HUFF_BIN_DIR'"
    rm -r $HUFF_BIN_DIR
fi

# Remove the CMake build directory.
if [ -d $HUFF_BUILD_DIR ]
then
    echo "removing '$HUFF_BUILD_DIR'"
    rm -r $HUFF_BUILD_DIR
fi
