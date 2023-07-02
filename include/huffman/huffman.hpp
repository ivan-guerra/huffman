#ifndef HUFFMAN_H_
#define HUFFMAN_H_

#include <memory>
#include <string>
#include <unordered_map>

namespace huffman {

enum class RetCode {
    kSuccess,
    kInvalidChar,
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

    RetCode Encode(const std::string& unarchived_filepath,
                   const std::string& archived_filepath);
    RetCode Decode(const std::string& archived_filepath,
                   const std::string& unarchived_filepath);

   private:
    struct HuffmanNode; /* forward decl the HuffmanNode type */
    using CharFreqMap = std::unordered_map<char, int>;
    using HuffmanNodePtr = std::shared_ptr<HuffmanNode>;

    static const int kInternalNode; /* special code for internal huffman node */

    struct HuffmanNode {
        int character;
        int count;
        HuffmanNodePtr zero;
        HuffmanNodePtr one;

        HuffmanNode(int character_, int count_, HuffmanNodePtr zero_ = nullptr,
                    HuffmanNodePtr one_ = nullptr)
            : character(character_), count(count_), zero(zero_), one(one_) {}
    };

    RetCode CountCharFrequencies(const std::string& filepath);
    void BuildEncodingTree();

    CharFreqMap char_freqs_;
    HuffmanNodePtr encoding_root_;
};

}  // namespace huffman

#endif
