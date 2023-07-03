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
    std::ifstream infile_stream(infile, std::ios::in);
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

    /* traverse the huffman tree decoding characters along the way */
    uint32_t num_chars_decoded = 0;
    std::size_t i = 0;
    HuffmanNodePtr node = encoding_root_;
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
    const std::size_t kBytesPerRead = 4096; /* size of one page */
    std::vector<char> buffer(kBytesPerRead, '\0');

    /* handle to read compressed data one page at a time */
    std::ifstream infile_stream(infile, std::ios::in | std::ios::binary);
    /* handle to write the uncompressed data */
    std::ofstream outfile_stream(outfile);

    RetCode retcode = ReadHeader(infile_stream); /* read in char frequencies */
    if (RetCode::kSuccess != retcode) {          /* invalid header */
        return retcode;
    }
    BuildEncodingTree(); /* construct the encoding tree */

    /* build up a bit vector from the compressed files bianry content */
    std::vector<bool> bitstream;
    while (infile_stream) {
        infile_stream.read(buffer.data(), buffer.size());
        for (std::streamsize i = 0; i < infile_stream.gcount(); ++i) {
            const int kNumBitsInByte = 8;
            for (int j = 0; j < kNumBitsInByte; ++j) {
                uint8_t mask = 1 << (kNumBitsInByte - j - 1);
                bitstream.push_back(buffer[i] & mask);
            }
        }
    }

    /* reconstruct the message by traversing the huffman tree */
    DecodeStream(bitstream, outfile_stream);

    return retcode;
}

RetCode HuffmanCoding::Compress(const std::string& unarchived_filepath,
                                const std::string& archived_filepath) {
    /* verify unarchived_filepath points to an existing file */
    std::filesystem::path unarchived_path(unarchived_filepath);
    if (!std::filesystem::exists(unarchived_filepath)) {
        return RetCode::kFileDoesNotExist;
    }

    /* scan the unarchived file once to compute ascii char frequencies */
    RetCode retcode = CountCharFrequencies(unarchived_filepath);
    if (RetCode::kSuccess != retcode) {
        return retcode;
    }

    BuildEncodingTree();                  /* construct the huffman code tree */
    BuildEncodingMap(encoding_root_, ""); /* construct char to encoding map */
    Encode(unarchived_filepath, archived_filepath); /* compress the data */

    return retcode;
}

RetCode HuffmanCoding::Decompress(const std::string& archived_filepath,
                                  const std::string& unarchived_filepath) {
    /* verify archived_filepath points to an existing file */
    std::filesystem::path archived_path(archived_filepath);
    if (!std::filesystem::exists(archived_filepath)) {
        return RetCode::kFileDoesNotExist;
    }

    return Decode(archived_filepath, unarchived_filepath);
}

}  // namespace huffman
