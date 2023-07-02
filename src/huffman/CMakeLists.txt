cmake_minimum_required(VERSION 3.13...3.25)

project(huffman_archiver DESCRIPTION "huffman coding"
                         LANGUAGES   CXX
)

add_library(${PROJECT_NAME} STATIC)

target_sources(${PROJECT_NAME}
    PRIVATE huffman.cc
)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${HUFF_INCLUDE_DIR}
)
