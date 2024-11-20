use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Compress(CompressArgs),
    Decompress(DecompressArgs),
}

#[derive(Args)]
#[command(about = "Compress data using Huffman compression.")]
struct CompressArgs {
    #[arg(help = "path to file to compress")]
    decompressed_data: std::path::PathBuf,

    #[arg(help = "path to save compressed data")]
    compressed_data: std::path::PathBuf,
}

#[derive(Args)]
#[command(about = "Decompress Huffman encoded data.")]
struct DecompressArgs {
    #[arg(help = "path to compressed data")]
    compressed_data: std::path::PathBuf,

    #[arg(help = "path to save decompressed data")]
    decompressed_data: std::path::PathBuf,
}

fn main() {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Compress(args) => {
            let config = huffman::CompressConfig::new(
                args.decompressed_data.clone(),
                args.compressed_data.clone(),
            );

            if let Err(e) = huffman::compress(&config) {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Decompress(args) => {
            let config = huffman::DecompressConfig::new(
                args.compressed_data.clone(),
                args.decompressed_data.clone(),
            );

            if let Err(e) = huffman::decompress(&config) {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        }
    }
}
