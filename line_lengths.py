import os
import collections

def analyze_line_lengths(directory=".", bucket_size=20):
    """
    Iterates over .py files in the given directory (recursively)
    and prints the distribution of line lengths bucketed.

    Args:
        directory (str): The directory to search in.
        bucket_size (int): The size of each line length bucket.
    """
    line_length_counts = collections.defaultdict(int)
    total_lines = 0
    total_py_files = 0

    print(f"Analyzing .py files in '{os.path.abspath(directory)}'...\n")

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                print(filename)
                total_py_files += 1
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            # Remove trailing newline character for length calculation
                            stripped_line_length = len(line.rstrip('\n'))
                            total_lines += 1

                            if stripped_line_length == 0:
                                # Handle 0-length lines specifically to fit the "0-20" bucket
                                # or assign to a special "empty lines" bucket if preferred.
                                # For this request, 0 length falls into 0-20.
                                bucket_idx = 0
                            else:
                                # (length - 1) // bucket_size ensures that a line of exactly
                                # bucket_size (e.g., 20) falls into the (0-20) bucket,
                                # and a line of length (bucket_size + 1) (e.g., 21)
                                # falls into the next bucket (20-40).
                                bucket_idx = (stripped_line_length - 1) // bucket_size

                            bucket_start = bucket_idx * bucket_size
                            bucket_end = bucket_start + bucket_size
                            bucket_label = f"{bucket_start}-{bucket_end} chars"
                            line_length_counts[bucket_label] += 1
                except Exception as e:
                    print(f"Could not read file {filepath}: {e}")

    if not line_length_counts:
        print("No .py files found or no lines in .py files.")
        return

    print("Line Length Distribution:")
    # Sort buckets numerically by their starting range
    # e.g., "0-20", "20-40", "100-120"
    sorted_buckets = sorted(
        line_length_counts.items(),
        key=lambda item: int(item[0].split('-')[0])
    )

    max_label_len = 0
    if sorted_buckets:
        max_label_len = max(len(label) for label, _ in sorted_buckets)

    for label, count in sorted_buckets:
        percentage = (count / total_lines) * 100 if total_lines > 0 else 0
        print(f"{label.ljust(max_label_len)}: {count:6d} ({percentage:6.2f}%)")

    print("-" * (max_label_len + 20))
    print(f"{'Total Python Files'.ljust(max_label_len)}: {total_py_files:6d}")
    print(f"{'Total Lines Scanned'.ljust(max_label_len)}: {total_lines:6d} (100.00%)")


if __name__ == "__main__":
    # Analyze the current directory by default
    # You can pass a different directory path if needed, e.g., analyze_line_lengths("/path/to/your/code")
    analyze_line_lengths()

    # Example with a different bucket size:
    # print("\n--- Analysis with bucket size 10 ---")
    # analyze_line_lengths(bucket_size=10)

