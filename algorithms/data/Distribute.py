import subprocess
import pickle


def distribute_process(init_args):
    # Serialize the class initialization arguments
    serialized_data = pickle.dumps({'init_args': init_args})

    # Run the worker script in a subprocess
    proc = subprocess.Popen(
        ['python3', 'worker_script.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Send serialized data to the subprocess
    stdout, stderr = proc.communicate(input=serialized_data)

    # Handle output or errors
    print("Output: ")
    print(stdout.decode())
    print("Errors: ")
    print(stderr.decode())


if __name__ == '__main__':
    length = 10
    file_path = '../../data/mssm/distributed_sampling/sample'

    for i in range(20):
        print("Process #", i)

        init_dict = {
            'n_samples': length,
            'file_path': file_path + str(i) + '.csv'
        }

        distribute_process(init_dict)
