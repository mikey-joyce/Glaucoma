import sys
import pickle

from Sample import Sample


def main():
    # Load the serialized data from command-line arguments
    serialized_data = sys.stdin.buffer.read()
    data = pickle.loads(serialized_data)

    # Instantiate the class and run the method
    instance = Sample(**data['init_args'])
    instance.save()

if __name__ == "__main__":
    main()

