import os
import ray
import time
import subprocess
import sys


def get_ray_path():
    # Run the command and capture its output
    result = subprocess.run(['which', 'ray'], capture_output=True, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        # Save the output to a variable
        output = result.stdout.strip()
        return output
    else:
        return None
    
user = os.path.expanduser('~')
ray_dir = get_ray_path()

with open("/var/emulab/boot/hostmap") as fp:
    N, *_ = fp.readlines()
N = int(N.rstrip())
head_ip, *worker_ips = ["10.1.1.%d" % (2+node_id) for node_id in list(range(N))]


head_port = 6379


def tear_down():
    if not ray_dir:
        print("Ray not found")
        return

    # stopping all previous instances
    for ip in [head_ip]+worker_ips:
        os.system(f"ssh {ip} {ray_dir} stop")


def start():


    if not ray_dir:
        print("Ray not found")
        return

    # stopping all previous instances
    for ip in [head_ip]+worker_ips:
        os.system(f"ssh {ip} {ray_dir} stop")

    # starting on head node
    os.system(f"ssh {head_ip} {ray_dir} start --head --node-ip-address={head_ip} --port={head_port} --redis-password=tf_cluster_123")


    for worker_ip in worker_ips:
        time.sleep(1)
        os.system(f"ssh {worker_ip} {ray_dir} start --address={head_ip}:{head_port} --redis-password=tf_cluster_123")


    time.sleep(5)


    ray.init(address="%s:%s" % (head_ip, head_port), _redis_password="tf_cluster_123")

    ray_nodes = list(filter(lambda n: n["alive"], ray.nodes()))
    print("Num of nodes: %d" % len(ray_nodes))
    print(ray.cluster_resources())
    print(ray.available_resources())


if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        start()
    else:
        globals()[sys.argv[1]]()