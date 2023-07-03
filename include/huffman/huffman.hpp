#ifndef HUFFMAN_H_
#define HUFFMAN_H_

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace huffman {

enum class RetCode {
    kSuccess,
    kInvalidChar,
    kInvalidFileFormat,
    kFileDoesNotExist,
};

class HuffmanCoding {
   public:
    HuffmanCoding() : encoding_root_(nullptr) {}
    ~HuffmanCoding() = default;

    HuffmanCoding(const HuffmanCoding&) = default;
    HuffmanCoding& operator=(const HuffmanCoding&) = default;
    HuffmanCoding(HuffmanCoding&&) = default;
    HuffmanCoding& operator=(HuffmanCoding&&) = default;

    RetCode Compress(const std::string& unarchived_filepath,
                     const std::string& archived_filepath);
    RetCode Decompress(const std::string& archived_filepath,
                       const std::string& unarchived_filepath);

   private:
    struct HuffmanNode; /* forward decl the HuffmanNode type */
    using CharFreqMap = std::map<char, uint32_t>;
    using EncodingMap = std::map<char, std::string>;
    using HuffmanNodePtr = std::shared_ptr<HuffmanNode>;

    static const int kInternalNode; /* special code for internal huffman node */

    struct HuffmanNode {
        int character;
        uint32_t count;
        HuffmanNodePtr zero;
        HuffmanNodePtr one;

        HuffmanNode(int character_, int count_, HuffmanNodePtr zero_ = nullptr,
                    HuffmanNodePtr one_ = nullptr)
            : character(character_), count(count_), zero(zero_), one(one_) {}
    };

    RetCode CountCharFrequencies(const std::string& filepath);
    void BuildEncodingTree();
    void BuildEncodingMap(HuffmanNodePtr root, std::string encoding);

    void WriteHeader(std::ofstream& os) const;
    void Encode(const std::string& infile, const std::string& outfile) const;

    RetCode ReadHeader(std::ifstream& is);
    void DecodeStream(const std::vector<bool>& bitstream, std::ofstream& os);
    RetCode Decode(const std::string& infile, const std::string& outfile);

    CharFreqMap char_freqs_;
    EncodingMap encodings_;
    HuffmanNodePtr encoding_root_;
};

}  // namespace huffman

#endif
