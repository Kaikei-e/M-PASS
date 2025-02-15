from learning_machine.learn import learn


def main(target_dir: str):
    model, history = learn(target_dir)
    print(history)
    model.save("model.h5")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 main.py <target_directory>")
        sys.exit(1)
    target_dir = sys.argv[1]
    main(target_dir)
