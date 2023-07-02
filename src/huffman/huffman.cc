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

void HuffmanCoding::WriteHeader(std::ofstream& os) const {
    std::size_t num_chars = char_freqs_.size();
    os.write(reinterpret_cast<char*>(&num_chars), sizeof(num_chars));
    for (const auto& [character, frequency] : char_freqs_) {
        os.write(&character, sizeof(character));

        uint32_t freq_copy = frequency;
        os.write(reinterpret_cast<char*>(&freq_copy), sizeof(freq_copy));
    }
}

void HuffmanCoding::Encode(const std::string& infile,
                           const std::string& outfile) const {
    const std::size_t kBytesPerRead = 4096; /* size of one page */
    std::vector<char> buffer(kBytesPerRead, '\0');

    /* handle to read uncompressed data one page at a time */
    std::ifstream infile_stream(infile, std::ios::in | std::ios::binary);
    /* handle to write the encoding map and data */
    std::ofstream outfile_stream(outfile, std::ios::out | std::ios::binary);

    /* controls for writing compressed data byte by byte */
    const int kBitsPerByte = 8;
    uint8_t currbyte = 0;
    int bitcount = 0;

    /* write compressed files' header first */
    WriteHeader(outfile_stream);

    while (infile_stream) {
        /* read uncompressed data */
        infile_stream.read(buffer.data(), buffer.size());

        /* encode the individual ascii chars in the buffer */
        for (std::streamsize i = 0; i < infile_stream.gcount(); ++i) {
            /* since the smallest unit we can write to a file is a byte not a
             * bit, the code below constructs a byte from the bits in an
             * encoding and then writes the byte to the output file */
            for (const char& bit : encodings_.at(buffer[i])) {
                uint8_t ibit = (bit == '1') ? 1 : 0;
                currbyte = (currbyte << 1) | ibit;
                bitcount++;
                if (bitcount == kBitsPerByte) {
                    outfile_stream.write(reinterpret_cast<char*>(&currbyte),
                                         sizeof(currbyte));
                    currbyte = 0;
                    bitcount = 0;
                }
            }
        }
    }

    if (bitcount) {
        /* the very last character didn't land on the byte boundary so we
        need to pad it with zeroes before writing it out to file */
        while (bitcount != kBitsPerByte) {
            currbyte <<= 1;
            bitcount++;
        }
        outfile_stream.write(reinterpret_cast<char*>(&currbyte),
                             sizeof(currbyte));
    }
}

RetCode HuffmanCoding::Compress(const std::string& unarchived_filepath,
                                const std::string& archived_filepath) {
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
    BuildEncodingMap(encoding_root_, ""); /* construct char to encoding map */
    Encode(unarchived_filepath, archived_filepath); /* compress the data */

    return RetCode::kSuccess;
}

RetCode HuffmanCoding::Decompress(const std::string& archived_filepath,
                                  const std::string& unarchived_filepath) {
    (void)archived_filepath;
    (void)unarchived_filepath;
    return RetCode::kSuccess;
}

}  // namespace huffman
