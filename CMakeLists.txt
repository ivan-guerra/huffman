cmake_minimum_required(VERSION 3.13...3.25)

project(huffman_coding DESCRIPTION "huffman coding")

set(HUFF_BIN_DIR "${CMAKE_SOURCE_DIR}/bin"
    CACHE STRING "${PROJECT_NAME} binary directory.")
set(HUFF_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include"
    CACHE STRING     "${PROJECT_NAME} include directory.")

add_subdirectory(docs)
add_subdirectory(src)
