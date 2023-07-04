#include <iostream>
#include <string>

#include "huffman/huffman.hpp"

void PrintUsage() {
    std::cout << "usage: archiver CMD IN_FILE OUT_FILE" << std::endl;
    std::cout << "\tCMD" << std::endl;
    std::cout << "\t\tone of 'compress', 'decompress', or 'help'" << std::endl;
    std::cout << "\tIN_FILE\n\t\tinput file to be compressed/decompressed"
              << std::endl;
    std::cout
        << "\tIN_FILE\n\t\toutput file storing compressed/decompressed data"
        << std::endl;
    std::cout << "EXAMPLES" << std::endl;
    std::cout << "\tarchiver compress data.txt data.huf" << std::endl;
    std::cout << "\tarchiver decompressed data.huf data.txt" << std::endl;
}

void PrintErrAndExit(const std::string& err) {
    std::cerr << "error: " << err << std::endl;
    std::cerr << "try 'archiver help' for more information" << std::endl;
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

    if (kRequiredCmdArgs != argc) { /* did we get the right cmd arg count */
        PrintErrAndExit("invalid arg count");
    }

    huffman::RetCode retcode = huffman::RetCode::kSuccess;
    huffman::HuffmanCoding archiver;
    if (kCompressCmd == cmd) {
        retcode = archiver.Compress(argv[2], argv[3]);
    } else if (kDecompressCmd == cmd) {
        retcode = archiver.Decompress(argv[2], argv[3]);
    } else if (kHelpCmd == cmd) {
        PrintUsage();
    }

    /* report errors if there are any */
    switch (retcode) {
        case huffman::RetCode::kSuccess:
            break;
        case huffman::RetCode::kInvalidChar:
            PrintErrAndExit(
                "non-ASCII char detected, only ASCII chars allowed");
            break;
        case huffman::RetCode::kInvalidFileFormat:
            PrintErrAndExit("invalid archive file format");
            break;
        case huffman::RetCode::kFileDoesNotExist:
            PrintErrAndExit("IN_FILE does not exist");
            break;
    }
    return 0;
}
