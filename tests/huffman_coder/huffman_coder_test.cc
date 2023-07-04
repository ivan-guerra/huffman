#include "huffman_coder/huffman_coder.hpp"

#include <gtest/gtest.h>

#include <string>

TEST(HuffmanCodingTests, CanCompressFile) {
    huffman::HuffmanCoding coder;
    const std::string kInputFile("data/valid.txt");
    const std::string kOutputFile("data/valid.huf");

    ASSERT_EQ(coder.Compress(kInputFile, kOutputFile),
              huffman::RetCode::kSuccess);
}

TEST(HuffmanCodingTests, CanDecompressEncodedData) {
    huffman::HuffmanCoding coder;
    const std::string kInputFile("data/valid.huf");
    const std::string kOutputFile("data/valid.txt");

    ASSERT_EQ(coder.Decompress(kInputFile, kOutputFile),
              huffman::RetCode::kSuccess);
}

TEST(HuffmanCodingTests, CannotCompressNonexistentInputFile) {
    huffman::HuffmanCoding coder;
    const std::string kNonexistentFile("foo.txt");
    const std::string kOutputFile("foo.huf");

    ASSERT_EQ(coder.Compress(kNonexistentFile, kOutputFile),
              huffman::RetCode::kFileDoesNotExist);
}

TEST(HuffmanCodingTests, CannotDecompressNonexistentInputFile) {
    huffman::HuffmanCoding coder;
    const std::string kNonexistentFile("foo.huf");
    const std::string kOutputFile("foo.txt");

    ASSERT_EQ(coder.Decompress(kNonexistentFile, kOutputFile),
              huffman::RetCode::kFileDoesNotExist);
}

TEST(HuffmanCodingTests, CannotCompressEmptyFile) {
    huffman::HuffmanCoding coder;
    const std::string kEmptyFile("data/empty.txt");
    const std::string kOutputFile("empty.huf");

    ASSERT_EQ(coder.Compress(kEmptyFile, kOutputFile),
              huffman::RetCode::kEmptyFile);
}

TEST(HuffmanCodingTests, CannotDecompressFileWithInvalidFormat) {
    huffman::HuffmanCoding coder;
    const std::string kInvalidFormat("data/invalid.huf");
    const std::string kOutputFile("invalid.txt");

    ASSERT_EQ(coder.Decompress(kInvalidFormat, kOutputFile),
              huffman::RetCode::kInvalidFileFormat);
}
