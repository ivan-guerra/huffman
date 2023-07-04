#include <iostream>
#include <string>

#include "huffman_coder/huffman_coder.hpp"

void PrintUsage() {
    std::cout << "usage: huffman CMD IN_FILE OUT_FILE" << std::endl;
    std::cout << "\tCMD" << std::endl;
    std::cout << "\t\tone of 'compress', 'decompress', or 'help'" << std::endl;
    std::cout << "\tIN_FILE\n\t\tinput file to be compressed/decompressed"
              << std::endl;
    std::cout
        << "\tIN_FILE\n\t\toutput file storing compressed/decompressed data"
        << std::endl;
    std::cout << "EXAMPLES" << std::endl;
    std::cout << "\thuffman compress data.txt data.huf" << std::endl;
    std::cout << "\thuffman decompressed data.huf data.txt" << std::endl;
}

void PrintErrAndExit(const std::string& err) {
    std::cerr << "error: " << err << std::endl;
    std::cerr << "try 'huffman help' for more information" << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
    const int kRequiredCmdArgs = 4;
    const std::string kCompressCmd("compress");
    const std::string kDecompressCmd("decompress");
    const std::string kHelpCmd("help");

    if (argc < 2) { /* missing the program command arg */
        PrintErrAndExit("missing command");
    }

    std::string cmd(argv[1]);
    if ((kCompressCmd != cmd) && (kDecompressCmd != cmd) &&
        (kHelpCmd != cmd)) { /* did we get a valid command */
        PrintErrAndExit("unknown CMD value");
    }

    if (kHelpCmd == cmd) { /* user just wants to see the help info */
        PrintUsage();
        return 0;
    }

    huffman::RetCode retcode = huffman::RetCode::kSuccess;
    huffman::HuffmanCoding coder;
    if (kRequiredCmdArgs == argc) { /* did we get the right cmd arg count */
        if (kCompressCmd == cmd) {
            retcode = coder.Compress(argv[2], argv[3]);
        } else if (kDecompressCmd == cmd) {
            retcode = coder.Decompress(argv[2], argv[3]);
        }
    } else {
        PrintErrAndExit("invalid arg count");
    }

    /* report errors if there are any */
    switch (retcode) {
        case huffman::RetCode::kSuccess:
            break;
        case huffman::RetCode::kInvalidFileFormat:
            PrintErrAndExit("invalid archive file format");
            break;
        case huffman::RetCode::kFileDoesNotExist:
            PrintErrAndExit("IN_FILE does not exist");
            break;
        case huffman::RetCode::kEmptyFile:
            PrintErrAndExit("IN_FILE is empty, nothing to compress");
            break;
    }
    return 0;
}
