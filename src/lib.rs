use bitvec::prelude::BitVec;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, Write};

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
        self.frequency == other.frequency && self.byte == other.byte
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
        // First compare by frequency in reverse order
        let freq_cmp = other.frequency.cmp(&self.frequency);
        if freq_cmp != Ordering::Equal {
            return freq_cmp;
        }

        // If frequencies are equal, compare bytes to ensure stable ordering
        match (self.byte, other.byte) {
            (Some(a), Some(b)) => a.cmp(&b),
            (None, None) => Ordering::Equal,
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
        }
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
    codebook: &mut HashMap<u8, BitVec>,
    bits: BitVec,
) {
    if let Some(byte) = huffman_tree.byte {
        codebook.insert(byte, bits);
        return;
    }

    if let Some(ref left) = huffman_tree.left {
        let mut left_bits = bits.clone();
        left_bits.push(false); // Add 0 for left path
        build_codebook(left, codebook, left_bits);
    }

    if let Some(ref right) = huffman_tree.right {
        let mut right_bits = bits.clone();
        right_bits.push(true); // Add 1 for right path
        build_codebook(right, codebook, right_bits);
    }
}

pub fn write_huffman_header<W: Write + Seek>(
    char_frequency_map: &HashMap<u8, u64>,
    output_file: &mut W,
) -> std::io::Result<()> {
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

pub fn read_huffman_header<W: Read + Seek>(
    input_file: &mut W,
) -> Result<HashMap<u8, u64>, Box<dyn Error>> {
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
    build_codebook(&huffman_tree, &mut codebook, BitVec::new());

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
            let code = codebook.get(&byte).unwrap();

            for bit in code.iter() {
                current_byte = (current_byte << 1) | if *bit { 1 } else { 0 };
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
    let num_chars = char_frequency_map.values().sum::<u64>();
    let mut num_chars_decoded = 0;
    let mut byte_index = 0;
    let mut current_byte = if !input_buffer.is_empty() {
        input_buffer[0]
    } else {
        0
    };
    let mut bits_remaining = 8;

    while num_chars_decoded < num_chars {
        let mut current_node = &huffman_tree;

        // Navigate the tree until we reach a leaf node
        while current_node.byte.is_none() {
            if byte_index >= input_buffer.len() {
                return Err("unexpected end of compressed data".into());
            }

            // Read the next bit
            let bit = (current_byte & (1 << (bits_remaining - 1))) != 0;

            current_node = if bit {
                current_node.right.as_ref().unwrap()
            } else {
                current_node.left.as_ref().unwrap()
            };

            // Move to next bit
            bits_remaining -= 1;
            if bits_remaining == 0 {
                byte_index += 1;
                if byte_index < input_buffer.len() {
                    current_byte = input_buffer[byte_index];
                }
                bits_remaining = 8;
            }
        }

        // Write the decoded byte
        output_file.write_all(&[current_node.byte.unwrap()])?;
        num_chars_decoded += 1;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Cursor;
    use testdir::testdir;

    #[test]
    fn load_char_frequency_map_can_read_basic_text() {
        let dir = testdir!();
        let test_file = dir.join("test.txt");
        fs::write(&test_file, "hello world").unwrap();

        let result = load_char_frequency_map(&test_file).unwrap();

        assert_eq!(result.get(&b'h').unwrap(), &1);
        assert_eq!(result.get(&b'e').unwrap(), &1);
        assert_eq!(result.get(&b'l').unwrap(), &3);
        assert_eq!(result.get(&b'o').unwrap(), &2);
        assert_eq!(result.get(&b'w').unwrap(), &1);
        assert_eq!(result.get(&b'r').unwrap(), &1);
        assert_eq!(result.get(&b'd').unwrap(), &1);
        assert_eq!(result.get(&b' ').unwrap(), &1);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn load_char_frequency_map_returns_empty_map_when_given_empty_file() {
        let dir = testdir!();
        let test_file = dir.join("empty.txt");
        fs::write(&test_file, "").unwrap();

        let result = load_char_frequency_map(&test_file).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn load_char_frequency_map_returns_error_when_given_nonexistent_file() {
        let dir = testdir!();
        let nonexistent = dir.join("nonexistent.txt");

        let result = load_char_frequency_map(&nonexistent);
        assert!(result.is_err());
    }

    #[test]
    fn load_char_frequency_map_returns_error_when_given_directory() {
        let dir = testdir!();

        let result = load_char_frequency_map(&dir.as_path().to_path_buf());
        assert!(result.is_err());
    }

    #[test]
    fn build_huffman_tree_returns_error_when_given_empty_char_freq_map() {
        let frequencies = HashMap::new();
        let tree = build_huffman_tree(&frequencies);
        assert!(tree.is_err());
    }

    #[test]
    fn build_huffman_tree_can_build_tree_with_single_node() {
        let mut frequencies = HashMap::new();
        frequencies.insert(b'a', 1);

        let tree = build_huffman_tree(&frequencies).unwrap();
        assert_eq!(tree.byte, Some(b'a'));
        assert_eq!(tree.frequency, 1);
        assert!(tree.left.is_none());
        assert!(tree.right.is_none());
    }

    #[test]
    fn build_huffman_tree_can_build_tree_with_multiple_nodes() {
        let mut frequencies = HashMap::new();
        frequencies.insert(b'a', 2);
        frequencies.insert(b'b', 3);
        frequencies.insert(b'c', 1);

        let tree = build_huffman_tree(&frequencies).unwrap();
        assert_eq!(tree.frequency, 6); // root should have total frequency
        assert!(tree.byte.is_none()); // root should not have a byte
    }

    #[test]
    fn build_huffman_tree_can_partition_chars() {
        let mut frequencies = HashMap::new();
        frequencies.insert(b'a', 5);
        frequencies.insert(b'b', 2);

        let tree = build_huffman_tree(&frequencies).unwrap();
        assert_eq!(tree.frequency, 7);
        assert!(tree.byte.is_none());

        // 'a' should be in the path with higher frequency
        let higher_freq_node =
            if tree.left.as_ref().unwrap().frequency > tree.right.as_ref().unwrap().frequency {
                tree.left.as_ref().unwrap()
            } else {
                tree.right.as_ref().unwrap()
            };
        assert_eq!(higher_freq_node.byte, Some(b'a'));
        assert_eq!(higher_freq_node.frequency, 5);
    }

    #[test]
    fn write_huffman_header_writes_basic_header() -> Result<(), std::io::Error> {
        // Create a test frequency map
        let mut char_frequency_map = HashMap::new();
        char_frequency_map.insert(b'a', 2);
        char_frequency_map.insert(b'b', 3);

        // Use a cursor as our "file" so we can write to memory
        let mut buffer = Cursor::new(Vec::new());

        // Write the header
        write_huffman_header(&char_frequency_map, &mut buffer)?;

        // Get the written bytes
        let written_data = buffer.into_inner();

        // Check magic bytes (HUFFMAN_MAGIC)
        assert_eq!(&written_data[0..4], HUFFMAN_MAGIC);

        // Check version
        assert_eq!(written_data[4], HUFFMAN_VERSION);

        // Check number of characters (2 in little endian)
        assert_eq!(&written_data[5..9], &2u32.to_le_bytes());

        // Extract both character entries
        let first_entry = &written_data[9..18];
        let second_entry = &written_data[18..27];

        // Create sets of expected entries
        let a_entry = [b'a']
            .iter()
            .chain(&2u64.to_le_bytes())
            .copied()
            .collect::<Vec<u8>>();
        let b_entry = [b'b']
            .iter()
            .chain(&3u64.to_le_bytes())
            .copied()
            .collect::<Vec<u8>>();

        // Check that both entries exist in either order
        assert!(
            (first_entry == a_entry.as_slice() && second_entry == b_entry.as_slice())
                || (first_entry == b_entry.as_slice() && second_entry == a_entry.as_slice())
        );

        // Verify total length
        assert_eq!(written_data.len(), 27); // 4 + 1 + 4 + (1 + 8) * 2

        Ok(())
    }

    #[test]
    fn write_huffman_header_writes_empty_map() -> Result<(), std::io::Error> {
        let char_frequency_map = HashMap::new();
        let mut buffer = Cursor::new(Vec::new());

        write_huffman_header(&char_frequency_map, &mut buffer)?;

        let written_data = buffer.into_inner();

        // Check magic bytes and version
        assert_eq!(&written_data[0..4], HUFFMAN_MAGIC);
        assert_eq!(written_data[4], HUFFMAN_VERSION);

        // Check number of characters (0 in little endian)
        assert_eq!(&written_data[5..9], &0u32.to_le_bytes());

        // Verify total length
        assert_eq!(written_data.len(), 9); // just magic + version + char count

        Ok(())
    }

    #[test]
    fn write_huffman_header_writes_map_with_max_char_frequency() -> Result<(), std::io::Error> {
        let mut char_frequency_map = HashMap::new();
        char_frequency_map.insert(b'x', u64::MAX); // Maximum possible frequency

        let mut buffer = Cursor::new(Vec::new());
        write_huffman_header(&char_frequency_map, &mut buffer)?;

        let written_data = buffer.into_inner();

        // Check the frequency bytes
        assert_eq!(&written_data[10..18], &u64::MAX.to_le_bytes());

        Ok(())
    }

    #[test]
    fn read_huffman_header_reads_basic_header() -> Result<(), Box<dyn Error>> {
        // Create test data
        let mut expected_frequencies = HashMap::new();
        expected_frequencies.insert(b'a', 2);
        expected_frequencies.insert(b'b', 3);

        // Create a buffer and write a valid header to it
        let mut write_buffer = Cursor::new(Vec::new());
        write_huffman_header(&expected_frequencies, &mut write_buffer)?;

        // Create a reader from the written data
        let buffer_contents = write_buffer.into_inner();
        let mut reader = Cursor::new(buffer_contents);

        // Read the header
        let read_frequencies = read_huffman_header(&mut reader)?;

        // Verify the frequencies match
        assert_eq!(read_frequencies, expected_frequencies);

        Ok(())
    }

    #[test]
    fn read_huffman_header_returns_error_on_invalid_magic_num() -> Result<(), Box<dyn Error>> {
        let mut invalid_data = vec![0, 1, 2, 3]; // Wrong magic bytes
        invalid_data.extend_from_slice(&[HUFFMAN_VERSION]); // Version
        invalid_data.extend_from_slice(&0u32.to_le_bytes()); // Empty frequency map

        let mut reader = Cursor::new(invalid_data);

        match read_huffman_header(&mut reader) {
            Err(e) => assert_eq!(e.to_string(), "invalid header magic"),
            Ok(_) => panic!("expected error for invalid magic bytes"),
        }

        Ok(())
    }

    #[test]
    fn read_huffman_header_returns_error_on_invalid_version_num() -> Result<(), Box<dyn Error>> {
        let mut invalid_data = Vec::new();
        invalid_data.extend_from_slice(&HUFFMAN_MAGIC); // Correct magic
        invalid_data.extend_from_slice(&[HUFFMAN_VERSION + 1]); // Wrong version
        invalid_data.extend_from_slice(&0u32.to_le_bytes()); // Empty frequency map

        let mut reader = Cursor::new(invalid_data);

        match read_huffman_header(&mut reader) {
            Err(e) => assert_eq!(e.to_string(), "unsupported header version"),
            Ok(_) => panic!("Expected error for invalid version"),
        }

        Ok(())
    }

    #[test]
    fn read_huffman_header_reads_empty_map() -> Result<(), Box<dyn Error>> {
        let empty_frequencies = HashMap::new();

        // Create a buffer and write an empty header
        let mut write_buffer = Cursor::new(Vec::new());
        write_huffman_header(&empty_frequencies, &mut write_buffer)?;

        // Read it back
        let buffer_contents = write_buffer.into_inner();
        let mut reader = Cursor::new(buffer_contents);

        let read_frequencies = read_huffman_header(&mut reader)?;

        assert_eq!(read_frequencies, empty_frequencies);

        Ok(())
    }

    #[test]
    fn read_huffman_header_reads_map_with_max_frequency() -> Result<(), Box<dyn Error>> {
        let mut frequencies = HashMap::new();
        frequencies.insert(b'x', u64::MAX);

        // Create a buffer and write header with max frequency
        let mut write_buffer = Cursor::new(Vec::new());
        write_huffman_header(&frequencies, &mut write_buffer)?;

        // Read it back
        let buffer_contents = write_buffer.into_inner();
        let mut reader = Cursor::new(buffer_contents);

        let read_frequencies = read_huffman_header(&mut reader)?;

        assert_eq!(read_frequencies.get(&b'x'), Some(&u64::MAX));

        Ok(())
    }

    #[test]
    fn read_huffman_header_returns_error_on_truncated_header() -> Result<(), Box<dyn Error>> {
        // Create valid data first
        let mut frequencies = HashMap::new();
        frequencies.insert(b'a', 1);

        let mut write_buffer = Cursor::new(Vec::new());
        write_huffman_header(&frequencies, &mut write_buffer)?;

        // Truncate the data
        let mut truncated_data = write_buffer.into_inner();
        truncated_data.truncate(truncated_data.len() - 1);

        let mut reader = Cursor::new(truncated_data);

        // Should fail with IO error
        assert!(read_huffman_header(&mut reader).is_err());

        Ok(())
    }
}
