cmake_minimum_required(VERSION 3.13...3.25)

project(huffman DESCRIPTION "huffman compression/decompression utility"
                LANGUAGES   CXX
)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE huffman.cc
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE huffman_coder
)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${HUFF_BIN_DIR}
)
