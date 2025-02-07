from pre_processor import processor
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process XML files.")
    parser.add_argument("input_file", type=str, help="Path to the input XML file")
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    args = parser.parse_args()

    processor.parse_and_split_xml(
        input_file=os.path.abspath(args.input_file),
        output_dir=os.path.abspath(args.output_dir),
        output_prefix="output_",
        split_func=processor.split_by_record_element,
    )
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 main.py <input_xml_file> <output_directory>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    # output_prefix は使用せず、グループごとに type属性そのもので出力する
    processor.parse_and_split_xml(input_file, output_dir, "", processor.split_by_record_element) 