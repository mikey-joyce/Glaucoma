import subprocess
import pickle
from concurrent.futures import ThreadPoolExecutor


def distribute_process(init_args, id):
    # Serialize the class initialization arguments
    serialized_data = pickle.dumps({'init_args': init_args})

    # Run the worker script in a subprocess
    proc = subprocess.Popen(
        ['python3', 'Worker.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Send serialized data to the subprocess
    stdout, stderr = proc.communicate(input=serialized_data)

    # Handle output or errors
    print(f"Output from process #{id}: ")
    print(stdout.decode())
    print(f"Errors from process #{id}: ")
    print(stderr.decode())


if __name__ == '__main__':
    length = 20
    file_path = '../../data/igps/distributed_sampling/sample'

    num_cores = 3

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for i in range(num_cores):
            print("Process #", i)

            init_dict = {
                'n_samples': length,
                'file_path': file_path + str(i) + '.csv'
            }

            futures.append(executor.submit(distribute_process, init_dict, i))
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Process failed with exception: {e}")
