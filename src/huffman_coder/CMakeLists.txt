cmake_minimum_required(VERSION 3.13...3.25)

project(huffman_coder DESCRIPTION "huffman encoder/decoder"
                      LANGUAGES   CXX
)

add_library(${PROJECT_NAME} STATIC)

target_sources(${PROJECT_NAME}
    PRIVATE huffman_coder.cc
)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${HUFF_INCLUDE_DIR}
)
