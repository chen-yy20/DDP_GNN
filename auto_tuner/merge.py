import json

def merge_trace_files(input_files, output_file):
    merged_trace = []

    # Iterate over each input file
    for file in input_files:
        with open(file, 'r') as f:
            # Load JSON data from each file
            trace_data = json.load(f)
            # Extend the merged_trace list with the events from each file
            merged_trace.extend(trace_data["traceEvents"])

    # Create a dictionary for the merged trace data
    merged_data = {
        "traceEvents": merged_trace,
        # Copy other fields such as "metadata", "displayTimeUnit", etc., if present in the input files
    }

    # Write the merged trace to the output file
    with open(output_file, 'w') as f:
        json.dump(merged_data, f)


if __name__ == '__main__':
    # List of input trace files to merge
    input_files = [
        "mixture_product_0.json",
        "mixture_product_1.json",
        "mixture_product_2.json"
    ]

    # Output file to store the merged trace
    output_file = "combine.json"

    # Call the merge_trace_files function
    merge_trace_files(input_files, output_file)
