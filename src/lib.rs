use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};

static HUFFMAN_MAGIC: [u8; 4] = [0x48, 0x55, 0x46, 0x46]; // "HUFF"
static HUFFMAN_VERSION: u8 = 1;

pub struct CompressConfig {
    pub decompressed_data: std::path::PathBuf,
    pub compressed_data: std::path::PathBuf,
}

impl CompressConfig {
    pub fn new(
        decompressed_data: std::path::PathBuf,
        compressed_data: std::path::PathBuf,
    ) -> CompressConfig {
        CompressConfig {
            decompressed_data,
            compressed_data,
        }
    }
}

pub struct DecompressConfig {
    pub compressed_data: std::path::PathBuf,
    pub decompressed_data: std::path::PathBuf,
}

impl DecompressConfig {
    pub fn new(
        compressed_data: std::path::PathBuf,
        decompressed_data: std::path::PathBuf,
    ) -> DecompressConfig {
        DecompressConfig {
            compressed_data,
            decompressed_data,
        }
    }
}

#[derive(Clone)]
pub struct HuffmanNode {
    pub byte: Option<u8>,
    pub frequency: u64,
    left: Option<Box<HuffmanNode>>,
    right: Option<Box<HuffmanNode>>,
}

impl HuffmanNode {
    pub fn new(
        byte: Option<u8>,
        frequency: u64,
        left: Option<Box<HuffmanNode>>,
        right: Option<Box<HuffmanNode>>,
    ) -> HuffmanNode {
        HuffmanNode {
            byte,
            frequency,
            left,
            right,
        }
    }
}

impl PartialEq for HuffmanNode {
    fn eq(&self, other: &Self) -> bool {
        self.frequency == other.frequency
    }
}

impl Eq for HuffmanNode {}

impl PartialOrd for HuffmanNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HuffmanNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering so higher frequencies have lower priority
        other.frequency.cmp(&self.frequency)
    }
}

pub fn load_char_frequency_map(
    decompressed_data: &std::path::PathBuf,
) -> Result<HashMap<u8, u64>, std::io::Error> {
    let mut char_frequency_map = HashMap::new();
    let mut file = File::open(decompressed_data)?;
    let mut buffer = [0; 4096]; // 4KB buffer

    loop {
        let bytes_read = file.read(&mut buffer[..])?;
        if bytes_read == 0 {
            break; // EOF reached
        }

        for &byte in &buffer[..bytes_read] {
            *char_frequency_map.entry(byte).or_insert(0) += 1;
        }
    }

    Ok(char_frequency_map)
}

pub fn build_huffman_tree(
    char_frequency_map: &HashMap<u8, u64>,
) -> Result<HuffmanNode, Box<dyn Error>> {
    let mut priority_queue = std::collections::BinaryHeap::new();

    if char_frequency_map.is_empty() {
        return Err("cannot build Huffman tree from empty input".into());
    }

    for (byte, frequency) in char_frequency_map {
        priority_queue.push(HuffmanNode::new(Some(*byte), *frequency, None, None));
    }

    while priority_queue.len() > 1 {
        let left_child = priority_queue
            .pop()
            .ok_or("failed to pop left child from priority queue")?;
        let right_child = priority_queue
            .pop()
            .ok_or("failed to pop right child from priority queue")?;

        let merged_node = HuffmanNode::new(
            None,
            left_child.frequency + right_child.frequency,
            Some(Box::new(left_child)),
            Some(Box::new(right_child)),
        );
        priority_queue.push(merged_node);
    }

    priority_queue
        .pop()
        .ok_or_else(|| "failed to extract final node from priority queue".into())
}

pub fn build_codebook(
    huffman_tree: &HuffmanNode,
    codebook: &mut HashMap<u8, (Vec<u8>, usize)>,
    mut bits: Vec<u8>,
    bit_length: usize,
) {
    if let Some(byte) = huffman_tree.byte {
        codebook.insert(byte, (bits, bit_length));
    } else {
        let byte_index = bit_length / 8;
        let bit_position = bit_length % 8;

        // Ensure we have enough space
        if byte_index >= bits.len() {
            bits.push(0);
        }

        if let Some(ref left) = huffman_tree.left {
            let left_bits = bits.clone();
            // Set bit to 0 (already 0 by default)
            build_codebook(left, codebook, left_bits, bit_length + 1);
        }

        if let Some(ref right) = huffman_tree.right {
            let mut right_bits = bits.clone();
            // Set bit to 1
            right_bits[byte_index] |= 1 << (7 - bit_position);
            build_codebook(right, codebook, right_bits, bit_length + 1);
        }
    }
}

pub fn write_huffman_header(
    char_frequency_map: &HashMap<u8, u64>,
    output_file: &mut File,
) -> Result<(), std::io::Error> {
    output_file.write_all(&HUFFMAN_MAGIC)?;
    output_file.write_all(&[HUFFMAN_VERSION])?;

    let num_chars = char_frequency_map.len() as u32;
    output_file.write_all(&num_chars.to_le_bytes())?;

    for (byte, frequency) in char_frequency_map {
        output_file.write_all(&[*byte])?;
        output_file.write_all(&frequency.to_le_bytes())?;
    }

    Ok(())
}

pub fn read_huffman_header(input_file: &mut File) -> Result<HashMap<u8, u64>, Box<dyn Error>> {
    let mut magic = [0; 4];
    input_file.read_exact(&mut magic)?;
    if magic != HUFFMAN_MAGIC {
        return Err("invalid header magic".into());
    }

    let mut version = [0; 1];
    input_file.read_exact(&mut version)?;
    if version[0] != HUFFMAN_VERSION {
        return Err("unsupported header version".into());
    }

    let mut num_chars = [0; 4];
    input_file.read_exact(&mut num_chars)?;
    let num_chars = u32::from_le_bytes(num_chars);

    let mut char_frequency_map = HashMap::new();
    for _ in 0..num_chars {
        let mut byte = [0; 1];
        input_file.read_exact(&mut byte)?;
        let byte = byte[0];

        let mut frequency = [0; 8];
        input_file.read_exact(&mut frequency)?;
        let frequency = u64::from_le_bytes(frequency);

        char_frequency_map.insert(byte, frequency);
    }

    Ok(char_frequency_map)
}

pub fn compress(config: &CompressConfig) -> Result<(), Box<dyn Error>> {
    // Create all the necessary data structures:
    //  - Load the character frequency map from the input file
    //  - Build the Huffman tree from the character frequency map
    //  - Build the codebook from the Huffman tree
    let char_frequency_map = load_char_frequency_map(&config.decompressed_data)?;
    let huffman_tree = build_huffman_tree(&char_frequency_map)?;
    let mut codebook = HashMap::new();
    build_codebook(&huffman_tree, &mut codebook, Vec::new(), 0);

    let mut input_file = File::open(&config.decompressed_data)?;
    let mut output_file = File::create(&config.compressed_data)?;
    let mut input_buffer = [0; 4096]; // 4KB input buffer
    let mut output_buffer = [0; 4096]; // 4KB output buffer
    let mut output_pos = 0;
    let mut current_byte = 0u8;
    let mut bit_count = 0;

    // Write the Huffman header to the output file
    write_huffman_header(&char_frequency_map, &mut output_file)?;

    loop {
        let bytes_read = input_file.read(&mut input_buffer[..])?;
        if bytes_read == 0 {
            break; // EOF reached
        }

        for &byte in &input_buffer[..bytes_read] {
            let (code, code_length) = codebook.get(&byte).unwrap();

            let mut bit_index = 0;
            while bit_index < *code_length {
                let code_byte = code[bit_index / 8];
                let bit = (code_byte >> (7 - (bit_index % 8))) & 1;

                current_byte = (current_byte << 1) | bit;
                bit_count += 1;

                if bit_count == 8 {
                    output_buffer[output_pos] = current_byte;
                    output_pos += 1;
                    current_byte = 0;
                    bit_count = 0;

                    if output_pos == output_buffer.len() {
                        output_file.write_all(&output_buffer)?;
                        output_pos = 0;
                    }
                }

                bit_index += 1;
            }
        }
    }

    // Handle any remaining bits in the final byte
    if bit_count > 0 {
        current_byte <<= 8 - bit_count;
        output_buffer[output_pos] = current_byte;
        output_pos += 1;
    }

    // Flush any remaining bytes in the output buffer
    if output_pos > 0 {
        output_file.write_all(&output_buffer[..output_pos])?;
    }

    Ok(())
}

pub fn decompress(config: &DecompressConfig) -> Result<(), Box<dyn Error>> {
    let mut input_file = File::open(&config.compressed_data)?;
    let char_frequency_map = read_huffman_header(&mut input_file)?;
    let huffman_tree = build_huffman_tree(&char_frequency_map)?;

    // Buffer the bytes (excluding the header) in a byte vector
    let mut input_buffer = Vec::new();
    input_file.read_to_end(&mut input_buffer)?;

    let mut output_file = File::create(&config.decompressed_data)?;
    let num_chars = char_frequency_map.len();
    let mut num_chars_decoded = 0;
    let mut i = 0;
    while (i < input_buffer.len()) && (num_chars_decoded < num_chars) {
        let current_byte = input_buffer[i];
        i += 1;
        let mut bit_index = 0;
        while bit_index < 8 {
            let mut current_node = &huffman_tree;
            while current_node.byte.is_none() {
                let bit = (current_byte >> (7 - bit_index)) & 1;
                bit_index += 1;

                if bit == 0 {
                    current_node = current_node.left.as_ref().unwrap();
                } else {
                    current_node = current_node.right.as_ref().unwrap();
                }
            }

            output_file.write_all(&[current_node.byte.unwrap()])?;
            num_chars_decoded += 1;

            if num_chars_decoded == num_chars {
                break;
            }
        }
    }

    Ok(())
}
