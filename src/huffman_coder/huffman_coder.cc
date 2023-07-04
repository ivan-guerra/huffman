#include "huffman_coder/huffman_coder.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace huffman {

const int HuffmanCoding::kReadBuffSize = 1024;
const int HuffmanCoding::kInternalNode = 256;

RetCode HuffmanCoding::CountCharFrequencies(const std::string& infile) {
    auto IsAscii = [](char c) {
        return !(static_cast<unsigned char>(c) > 127);
    };

    /* read the input file in kReadBuffSize sized chunks */
    std::ifstream infile_stream(infile, std::ios::binary);
    while (infile_stream) {
        infile_stream.read(read_buffer_.data(), read_buffer_.size());
        for (std::streamsize i = 0; i < infile_stream.gcount(); ++i) {
            if (!IsAscii(read_buffer_[i])) { /* we don't allow non-ascii
                                                characters */
                return RetCode::kInvalidChar;
            }
            char_freqs_[read_buffer_[i]]++; /* up the char's frequency */
        }
    }
    return (char_freqs_.empty()) ? RetCode::kEmptyFile : RetCode::kSuccess;
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
    if (root->character != kInternalNode) { /* reached a leaf node */
        encodings_[static_cast<char>(root->character)] = encoding;
        return;
    }
    BuildEncodingMap(root->zero, encoding + "0"); /* recurse into ltree */
    BuildEncodingMap(root->one, encoding + "1");  /* recurse into rtree */
}

void HuffmanCoding::WriteHeader(std::ofstream& os) const {
    /* serialization of the header is as simple as it gets, we write the
     * number of key/value pairs in the char_freqs_ map followed by each
     * key/value itself */
    std::size_t num_chars = char_freqs_.size();
    os.write(reinterpret_cast<char*>(&num_chars), sizeof(num_chars));
    for (const auto& [character, frequency] : char_freqs_) {
        os.write(&character, sizeof(character));
        os.write(reinterpret_cast<const char*>(&frequency), sizeof(frequency));
    }
}

void HuffmanCoding::Encode(const std::string& infile,
                           const std::string& outfile) {
    /* controls for writing compressed data byte by byte */
    const int kBitsPerByte = 8;
    uint8_t currbyte = 0;
    int bitcount = 0;

    std::ofstream outfile_stream(outfile, std::ios::out | std::ios::binary);
    WriteHeader(outfile_stream); /* write the compressed files' header first */

    std::ifstream infile_stream(infile, std::ios::in);
    while (infile_stream) {
        /* read uncompressed data */
        infile_stream.read(read_buffer_.data(), read_buffer_.size());

        /* encode the individual ascii chars in the buffer */
        for (std::streamsize i = 0; i < infile_stream.gcount(); ++i) {
            /* since the smallest unit we can write to a file is a byte not a
             * bit, the code below constructs a byte from the bits in an
             * encoding and then writes the byte to the output file */
            for (const char& bit : encodings_.at(read_buffer_[i])) {
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

    if (bitcount) { /* the very last character didn't land on the byte boundary
                       so we need to pad it with zeroes before writing it out to
                       file */
        while (bitcount != kBitsPerByte) {
            currbyte <<= 1;
            bitcount++;
        }
        outfile_stream.write(reinterpret_cast<char*>(&currbyte),
                             sizeof(currbyte));
    }
}

RetCode HuffmanCoding::ReadHeader(std::ifstream& is) {
    if (!is) { /* missing header content */
        return RetCode::kInvalidFileFormat;
    }

    /* get the number of entries in the char frequency table */
    std::size_t num_chars = 0;
    is.read(reinterpret_cast<char*>(&num_chars), sizeof(num_chars));

    char character = '\0';
    uint32_t frequency = 0;
    for (std::size_t i = 0; i < num_chars; ++i) {
        if (!is) { /* header is missing one or more table entries */
            return RetCode::kInvalidFileFormat;
        }

        /* load the key (char) and value (frequency) */
        is.read(&character, sizeof(character));
        is.read(reinterpret_cast<char*>(&frequency), sizeof(frequency));

        char_freqs_[character] = frequency; /* add the new entry */
    }
    return RetCode::kSuccess;
}

void HuffmanCoding::DecodeStream(const std::vector<bool>& bitstream,
                                 std::ofstream& os) {
    /* take a tally of how many chars we need to decode */
    uint32_t num_chars = 0;
    for (const auto& kv : char_freqs_) {
        num_chars += kv.second;
    }

    /* repeatedly traverse the huffman tree decoding characters along the way */
    uint32_t num_chars_decoded = 0;
    HuffmanNodePtr node = encoding_root_;
    std::size_t i = 0;
    while ((i < bitstream.size()) && (num_chars_decoded != num_chars)) {
        node = (bitstream[i]) ? node->one : node->zero;

        if (!node->zero && !node->one) { /* reached a leaf node */
            os << static_cast<char>(node->character);
            node = encoding_root_;
            num_chars_decoded++;
        }
        i++;
    }
}

RetCode HuffmanCoding::Decode(const std::string& infile,
                              const std::string& outfile) {
    std::ifstream infile_stream(infile, std::ios::in | std::ios::binary);
    RetCode retcode = ReadHeader(infile_stream); /* read in char frequencies */
    if (RetCode::kSuccess != retcode) {          /* invalid header */
        return retcode;
    }

    BuildEncodingTree(); /* construct the encoding tree */

    /* build up a bit vector from the compressed file's binary content */
    const int kNumBitsInByte = 8;
    std::vector<bool> bitstream;
    while (infile_stream) {
        infile_stream.read(read_buffer_.data(), read_buffer_.size());
        for (std::streamsize i = 0; i < infile_stream.gcount(); ++i) {
            for (int j = 0; j < kNumBitsInByte; ++j) {
                uint8_t mask = 1 << (kNumBitsInByte - j - 1);
                bitstream.push_back(read_buffer_[i] & mask);
            }
        }
    }

    /* reconstruct the message by traversing the huffman tree */
    std::ofstream outfile_stream(outfile);
    DecodeStream(bitstream, outfile_stream);

    return retcode;
}

RetCode HuffmanCoding::Compress(const std::string& uncompressed_filepath,
                                const std::string& compressed_filepath) {
    /* verify uncompressed_filepath points to an existing file */
    std::filesystem::path uncompressed_path(uncompressed_filepath);
    if (!std::filesystem::exists(uncompressed_filepath)) {
        return RetCode::kFileDoesNotExist;
    }

    /* scan the uncompressed file once to compute ascii char frequencies */
    RetCode retcode = CountCharFrequencies(uncompressed_filepath);
    if (RetCode::kSuccess != retcode) {
        return retcode;
    }

    BuildEncodingTree();                  /* construct the huffman code tree */
    BuildEncodingMap(encoding_root_, ""); /* construct char to bit string map */
    Encode(uncompressed_filepath, compressed_filepath); /* compress the data */

    return retcode;
}

RetCode HuffmanCoding::Decompress(const std::string& compressed_filepath,
                                  const std::string& uncompressed_filepath) {
    /* verify compressed_filepath points to an existing file */
    std::filesystem::path compressed_path(compressed_filepath);
    if (!std::filesystem::exists(compressed_filepath)) {
        return RetCode::kFileDoesNotExist;
    }

    return Decode(compressed_filepath, uncompressed_filepath);
}

}  // namespace huffman
