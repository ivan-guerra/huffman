#ifndef HUFFMAN_H_
#define HUFFMAN_H_

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
    HuffmanCoding() = default;
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
    using CharFreqMap = std::unordered_map<char, int>;

    RetCode CountCharFrequencies(const std::string& filepath);

    CharFreqMap char_freqs_;
};

}  // namespace huffman

#endif
