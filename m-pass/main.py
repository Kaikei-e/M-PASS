import argparse
from learning_machine.laern import learn


def main():
    parser = argparse.ArgumentParser(description="Process XML files.")
    parser.add_argument("target_dir", type=str, help="Path to the target directory")
    args = parser.parse_args()
    print("target_dir: ", args.target_dir)
    learn(args.target_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 main.py <target_directory>")
        sys.exit(1)
    target_dir = sys.argv[1]
    main()
