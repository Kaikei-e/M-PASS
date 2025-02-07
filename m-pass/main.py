import argparse
from learning_machine.laern import learn


def main(target_dir: str):
    learn(target_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 main.py <target_directory>")
        sys.exit(1)
    target_dir = sys.argv[1]
    main(target_dir)
