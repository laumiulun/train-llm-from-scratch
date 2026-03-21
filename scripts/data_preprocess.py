import os
import json
import zstandard as zstd
import tiktoken
import h5py
from tqdm import tqdm
import argparse
import pathlib
import time
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple

def process_files(input_dir: str, output_file: str, tokenizer_name: str, max_data_percent: Optional[int] = None) -> None:
    """
    Process a specified number of lines from each .jsonl.zst file in the input directory
    and save encoded tokens to an HDF5 file.

    Args:
        input_dir (str): Directory containing input .jsonl.zst files.
        output_file (str): Path to the output HDF5 file.
        tokenizer_name (str): Name of the tiktoken tokenizer to use (e.g., 'r50k_base').
        max_data (int, optional): Maximum number of lines to process from each file.
                                  If None, process all lines.
    """
    # Print processing strategy based on max_data
    if max_data_percent is not None:
        print(f"You have chosen max_data_percent = {max_data_percent*100}%. Processing only the top {max_data_percent*100} % of JSON objects from each file.")
    else:
        print("Processing all available JSON objects from each file.")

    # Load the tokenizer using the provided tokenizer name
    enc = tiktoken.get_encoding(tokenizer_name)

    # Create an HDF5 file for output
    with h5py.File(output_file, 'w') as out_f:
        # Initialize the dataset for storing tokenized data
        dataset = out_f.create_dataset('tokens', (0,), maxshape=(None,), dtype='i')
        start_index = 0  # Track the starting index for the next batch of tokens

        # Process each .jsonl.zst file in the input directory
        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith(".jsonl.zst"):  # Only process .jsonl.zst files
                in_file = os.path.join(input_dir, filename)
                print(f"Processing: {in_file}")

                processed_lines = 0  # Counter for processed lines in the current file
                count = 0
                with zstd.open(in_file,'rt',encoding='utf-8') as f:
                    count += sum(1 for _ in f)

                max_data = int(max_data_percent * count)
                print(f"Using {max_data} rows from {in_file}")
                # Open the compressed .jsonl.zst file for reading
                with zstd.open(in_file, 'rt', encoding='utf-8') as in_f:
                    # Iterate over each line in the file
                    for line in tqdm(in_f, desc=f"Processing {filename}", total=max_data if max_data is not None else None):
                        try:
                            # Parse the line as JSON
                            data = json.loads(line)
                            text = data.get('text')  # Extract the 'text' field from the JSON object

                            if text:
                                # Tokenize the text and append an end-of-text token
                                encoded = enc.encode(text + "<|endoftext|>", allowed_special={'<|endoftext|>'})
                                encoded_len = len(encoded)

                                # Resize the dataset to accommodate new tokens
                                end_index = start_index + encoded_len
                                dataset.resize(dataset.shape[0] + encoded_len, axis=0)

                                # Store the encoded tokens in the dataset
                                dataset[start_index:end_index] = encoded
                                start_index = end_index  # Update the start index
                            else:
                                # Warn if 'text' key is missing in the JSON object
                                print(f"Warning: 'text' key missing in line from {filename}")
                        except json.JSONDecodeError:
                            # Handle JSON decoding errors
                            print(f"Warning: Could not decode JSON from line in {filename}")
                        except Exception as e:
                            # Handle any other errors
                            print(f"An error occurred while processing line in {filename}: {e}")

                        processed_lines += 1
                        # Stop processing if max_data limit is reached
                        if max_data is not None and processed_lines >= max_data:
                            break

def process_single_file(args: Tuple[str, str, str, float]) -> int:
    """
    Worker function to process a single .jsonl.zst file.
    """
    in_file, tokenizer_name, temp_h5_path, max_data_percent = args
    enc = tiktoken.get_encoding(tokenizer_name)
    tokens_count = 0

    # First pass: count lines to determine limit
    with zstd.open(in_file, 'rt', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    max_data = int(max_data_percent * total_lines)

    # Second pass: process and store in a temporary per-process H5 file
    with zstd.open(in_file, 'rt', encoding='utf-8') as in_f, \
         h5py.File(temp_h5_path, 'w') as out_f:

        dataset = out_f.create_dataset('tokens', (0,), maxshape=(None,), dtype='i')
        start_index = 0

        for i, line in enumerate(in_f):
            if i >= max_data:
                break
            try:
                data = json.loads(line)
                text = data.get('text')
                if text:
                    encoded = enc.encode(text + "<|endoftext|>", allowed_special={'<|endoftext|>'})
                    encoded_len = len(encoded)
                    dataset.resize(dataset.shape[0] + encoded_len, axis=0)
                    dataset[start_index:start_index + encoded_len] = encoded
                    start_index += encoded_len
            except (json.JSONDecodeError, Exception):
                continue

        tokens_count = dataset.shape[0]

    return tokens_count


def process_directory_multiprocess(input_dir: str, output_file: str, tokenizer_name: str, max_data_percent: float, num_workers: int):
    """
    Orchestrates the multiprocessing task and merges temporary results.
    """
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jsonl.zst")])
    if not files:
        return

    # Create arguments for pool
    temp_files = [f"{output_file}.part{i}.h5" for i in range(len(files))]
    pool_args = [(f, tokenizer_name, tf, max_data_percent) for f, tf in zip(files, temp_files)]

    print(f"Starting parallel processing with {num_workers} workers...")
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_file, pool_args), total=len(files), desc="Processing files"))

    # Final Step: Merge all temporary H5 files into one
    print("Merging temporary files...")
    with h5py.File(output_file, 'w') as final_f:
        total_tokens = sum(results)
        final_dataset = final_f.create_dataset('tokens', (total_tokens,), dtype='i')

        current_pos = 0
        for temp_f_path in tqdm(temp_files, desc="Merging"):
            with h5py.File(temp_f_path, 'r') as temp_f:
                data = temp_f['tokens'][:]
                final_dataset[current_pos : current_pos + len(data)] = data
                current_pos += len(data)
            os.remove(temp_f_path) # Clean up temp files

def main():
    """
    Main function to parse arguments, validate directories, and process files.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess PILE dataset files and save tokens to HDF5.")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Directory containing training .jsonl.zst files.")
    parser.add_argument("--val_dir", type=str, default="data/val", help="Directory containing validation .jsonl.zst files.")
    parser.add_argument("--out_train_file", type=str, default="data/train/pile_train.h5", help="Path to the output training HDF5 file.")
    parser.add_argument("--out_val_file", type=str, default="data/val/pile_dev.h5", help="Path to the output validation HDF5 file.")
    parser.add_argument("--tokenizer_name", type=str, default="r50k_base", help="Name of the tiktoken tokenizer to use.")
    parser.add_argument("--max_data_percent", type=float, default=0.01, help="Maximum percentage of json objects to process from each file in both train and val datasets (default: 1000).")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel processes.")

    args = parser.parse_args()

    for d, out in [(args.train_dir, args.out_train_file), (args.val_dir, args.out_val_file)]:
        if os.path.isdir(d):
            print(f"Processing directory: {d}")
            process_directory_multiprocess(d, out, args.tokenizer_name, args.max_data_percent, args.workers)

# Entry point of the script
if __name__ == "__main__":
    train_dir = "data/train"
    val_dir = "data/val"
    tokenizer_name = "r50k_base"
    max_percent = 6
    for i in range(1,max_percent + 1):
        max_data_percent = round(i*0.1,2)
        train_out = f"data/train/pile_train_{str(max_data_percent).replace(".","")}.h5"
        val_out = f"data/val/pile_dev_{str(max_data_percent).replace(".","")}.h5"

        if pathlib.Path(train_out).exists() and pathlib.Path(val_out).exists():
            print(f"Path {train_out} exists already, skipping")
        else:
            prev_time = time.time()
            for d, out in [(train_dir, train_out), (val_dir, val_out)]:
                if os.path.isdir(d):
                    print(f"Processing directory: {d}")
                    process_directory_multiprocess(d, out, tokenizer_name, max_data_percent, 15)
            print(f"Time it took: {time.time() - prev_time}")
        # print(val,train_out,val_out)

        # process_files(train_dir,train_out,tokenizer_name,val)
        # print("Training data preprocessing complete.")

        # process_files(val_dir,val_out,tokenizer_name,val)
        # print("Validation data preprocessing complete.")
