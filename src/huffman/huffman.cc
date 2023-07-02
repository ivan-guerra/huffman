#include "huffman/huffman.hpp"

#include <filesystem>
#include <fstream>
#include <queue>
#include <string>
#include <vector>

namespace huffman {

const int HuffmanCoding::kInternalNode = 256;

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

void HuffmanCoding::BuildEncodingTree() {
    auto HuffmanNodePtrGreater = [](const HuffmanNodePtr a,
                                    const HuffmanNodePtr b) {
        return (a->count > b->count);
    };
    std::priority_queue<HuffmanNodePtr, std::vector<HuffmanNodePtr>,
                        decltype(HuffmanNodePtrGreater)>
        encoding_queue;

    /* load the initial nodes with their chars and freqs */
    for (const auto& [character, frequency] : char_freqs_) {
        encoding_queue.push(
            std::make_shared<HuffmanNode>(character, frequency));
    }

    /* follow the algorithm described in
     * https://en.wikipedia.org/wiki/Huffman_coding under the "Compression"
     * section */
    while (encoding_queue.size() != 1) {
        HuffmanNodePtr first = encoding_queue.top();
        encoding_queue.pop();
        HuffmanNodePtr second = encoding_queue.top();
        encoding_queue.pop();

        HuffmanNodePtr new_node = std::make_shared<HuffmanNode>(
            kInternalNode, first->count + second->count, first, second);

        encoding_queue.push(new_node);
    }

    encoding_root_ = encoding_queue.top(); /* save off the root of the tree */
}

void HuffmanCoding::BuildEncodingMap(HuffmanNodePtr root,
                                     std::string encoding) {
    if (root->character != kInternalNode) {
        encodings_[static_cast<char>(root->character)] = encoding;
        return;
    }
    BuildEncodingMap(root->zero, encoding + "0");
    BuildEncodingMap(root->one, encoding + "1");
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

    BuildEncodingTree();                  /* construct the huffman code tree */
    BuildEncodingMap(encoding_root_, ""); /* generate char to code map */

    return RetCode::kSuccess;
}

RetCode HuffmanCoding::Decode(const std::string& archived_filepath,
                              const std::string& unarchived_filepath) {
    (void)archived_filepath;
    (void)unarchived_filepath;
    return RetCode::kSuccess;
}

}  // namespace huffman
