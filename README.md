# Huffman Coding

This project implements a command line utility that compresses and decompresses
any kind of file (text, image, audio, etc.) using the [Huffman Coding][1]
technique.

### Program Usage

The `huffman` tool interprets two commands: `compress` and `decompress`.

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

Run `huffman --help` for more information.

[1]: https://en.wikipedia.org/wiki/Huffman_coding
