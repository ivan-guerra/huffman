cmake_minimum_required(VERSION 3.13...3.25)

project(archiver DESCRIPTION "archive utility using huffman coding"
                 LANGUAGES   CXX
)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE archiver.cc
)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${HUFF_BIN_DIR}
)
