#ifndef HUFFMAN_H_
#define HUFFMAN_H_

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace huffman {

/** Success and failure return codes. */
enum class RetCode {
    kSuccess,           /**< The operation succeeded. */
    kInvalidChar,       /**< A non-ASCII character was encountered. */
    kInvalidFileFormat, /**< The compressed file format was incorrect. */
    kFileDoesNotExist,  /**< The specified file does not exist. */
};

/**
 * Compress/decompress data using the huffman coding technique.
 *
 * HuffmanCoding implements the algorithms described in the
 * <a href="https://en.wikipedia.org/wiki/Huffman_coding">Huffman Coding</a>
 * Wikipedia page.
 */
class HuffmanCoding {
   public:
    HuffmanCoding()
        : encoding_root_(nullptr), read_buffer_(kReadBuffSize, '\0') {}
    ~HuffmanCoding() = default;
    HuffmanCoding(const HuffmanCoding&) = default;
    HuffmanCoding& operator=(const HuffmanCoding&) = default;
    HuffmanCoding(HuffmanCoding&&) = default;
    HuffmanCoding& operator=(HuffmanCoding&&) = default;

    /**
     * Compress the contents of an ASCII data file.
     *
     * @param [in] unarchived_filepath Path to the file to be compressed.
     * @param [in] archived_filepath   Path to file that will contain compressed
     *                                 data.
     * @returns See #RetCode.
     */
    RetCode Compress(const std::string& unarchived_filepath,
                     const std::string& archived_filepath);

    /**
     * Decompress an archive file previously compressed via Compress().
     *
     * @param [in] archived_filepath   Path to a huffman encoded archive file.
     * @param [in] unarchived_filepath Path to file that will contain
     *                                 uncompressed data.
     * @returns See #RetCode.
     */
    RetCode Decompress(const std::string& archived_filepath,
                       const std::string& unarchived_filepath);

   private:
    using CharFreqMap = std::map<char, uint32_t>;
    using EncodingMap = std::map<char, std::string>;
    struct HuffmanNode;
    using HuffmanNodePtr = std::shared_ptr<HuffmanNode>;

    static const int kReadBuffSize; /**< Input file read buffer size. */
    static const int kInternalNode; /**< Char code for an internal node. */

    struct HuffmanNode {
        int character;       /**< ASCII character or kInternalNode value. */
        uint32_t count;      /**< Character frequency. */
        HuffmanNodePtr zero; /**< Huffman tree left subtree. */
        HuffmanNodePtr one;  /**< Huffman tree right subtree. */

        HuffmanNode(int character_, int count_, HuffmanNodePtr zero_ = nullptr,
                    HuffmanNodePtr one_ = nullptr)
            : character(character_), count(count_), zero(zero_), one(one_) {}
    };

    /**
     * Initialize the \a char_freqs_ character frequency map.
     *
     * @param filepath Path to an ASCII text file.
     *
     * @returns See #RetCode.
     */
    RetCode CountCharFrequencies(const std::string& filepath);

    /**
     * Initialize \a encoding_root_ to contain the huffman coding tree.
     *
     * We use the priority queue based algorithm described in the "Compression"
     * section of the Huffman Coding Wikipedia page to build up the tree.
     */
    void BuildEncodingTree();

    /** Initialize \a encodings_ to map characters to binary strings. */
    void BuildEncodingMap(HuffmanNodePtr root, std::string encoding);

    /** Serialize and write Huffman encoding data to an output stream. */
    void WriteHeader(std::ofstream& os) const;

    /** Encode the ASCII data in \p infile to the output file \p outfile. */
    void Encode(const std::string& infile, const std::string& outfile);

    /** Deserialize Huffman encoding data from an archive file stream. */
    RetCode ReadHeader(std::ifstream& is);

    /**
     * Decode and output encoded characters represented in \p bitstream.
     *
     * @param [in] bitstream A boolean vector representing the individual bits
     *                       read from an encoded data stream.
     * @param [in] os        An output stream object used to write decoded
     *                       ASCII character data.
     */
    void DecodeStream(const std::vector<bool>& bitstream, std::ofstream& os);

    /** Decode compressed data in \p infile to the output file \p outfile. */
    RetCode Decode(const std::string& infile, const std::string& outfile);

    CharFreqMap char_freqs_; /**< Map of character frequencies in the input. */
    EncodingMap encodings_; /**< Map of character to binary string encodings. */
    HuffmanNodePtr encoding_root_;  /**< Binary tree repr of Huffman codes. */
    std::vector<char> read_buffer_; /**< Holds data read from an input file. */
};

}  // namespace huffman

#endif
