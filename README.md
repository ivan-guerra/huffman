# Huffman Coding

This project implements a command line utility that compresses and decompresses
any kind of file using the [Huffman Coding][1] technique.

### Building

To build the project locally, you will need the following libraries and tools
installed:

* CMake3.13+
* C++ compiler supporting C++20 features

To build the project, change directory to the `scripts/` directory and run
`build.sh`

```bash
./build.sh
```
> **Note**
> `build.sh` supports additional option flags for building the project docs
> (requires [Doxygen][2]) and unit tests. Run `build.sh -h` to see all the
> options.

After a successful build, you will find the binary installed to
`huffman/bin/huffman`.

### Program Usage

The `huffman` tool interprets three commands: `help`, `compress`, and
`decompress`.

The `help` command prints program usage info:

```bash
huffman help
```

The `compress` command takes two arguments where the first is the file to be
compressed and the second argument is the name of the output file:

```bash
huffman compress foo.txt foo.huf
```

The `decompress` command takes two arguments where the first argument is a file
previously compressed by the `huffman` utility and the second argument is the
name of the decompressed file:

```bash
huffman decompress foo.huf foo.txt
```

[1]: https://en.wikipedia.org/wiki/Huffman_coding
[2]: https://www.doxygen.nl/
