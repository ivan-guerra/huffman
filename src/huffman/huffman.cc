#include "huffman/huffman.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace huffman {

RetCode HuffmanCoding::CountCharFrequencies(const std::string& infile) {
    auto IsAscii = [](char c) {
        return !(static_cast<unsigned char>(c) > 127);
    };

    std::ifstream infile_stream(infile, std::ios::binary);
    const std::size_t kBytesPerRead = 4096; /* size of one page */
    std::vector<char> buffer(kBytesPerRead, '\0');

    /* read the input file in page sized chunks */
    while (infile_stream) {
        infile_stream.read(buffer.data(), buffer.size());
        for (std::streamsize i = 0; i < infile_stream.gcount(); ++i) {
            if (!IsAscii(buffer[i])) { /* we don't allow non-ascii characters */
                return RetCode::kInvalidChar;
            }
            char_freqs_[buffer[i]]++; /* up the char's frequency */
        }
    }
    return RetCode::kSuccess;
}

RetCode HuffmanCoding::Encode(const std::string& unarchived_filepath,
                              const std::string& archived_filepath) {
    (void)archived_filepath;

    /* verify unarchived_filepath points to an existing file */
    std::filesystem::path unarchived_path(unarchived_filepath);
    if (!std::filesystem::exists(unarchived_filepath)) {
        return RetCode::kFileDoesNotExist;
    }

    /* scan the unarchived file once to compute ascii char frequencies */
    RetCode rc = CountCharFrequencies(unarchived_filepath);
    if (RetCode::kSuccess != rc) {
        return rc;
    }

    return RetCode::kSuccess;
}

RetCode HuffmanCoding::Decode(const std::string& archived_filepath,
                              const std::string& unarchived_filepath) {
    (void)archived_filepath;
    (void)unarchived_filepath;
    return RetCode::kSuccess;
}

}  // namespace huffman
