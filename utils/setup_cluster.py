import os
import sys
from threading import Thread
import time
import subprocess
import fileinput

user = os.path.expanduser('~')
user_path = os.path.expanduser('~')
user_name = user_path.split('/')[-1]


with open("/var/emulab/boot/hostmap") as fp:
    N, *_ = fp.readlines()
N = int(N.rstrip())
list_of_nodes = ["10.1.1.%d" % (2+node_id) for node_id in list(range(N))]

head_node, *worker_nodes = list_of_nodes[:]


def setup_keys():
    def check_ssh(ip):
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', ip, 'exit'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0


    os.system(f'bash setup_keys.sh {N} {user_name}')
    time.sleep(1)
    
    for nodes in list_of_nodes:
        if check_ssh(nodes):
            print(f"SSH connection to {nodes} successful")
        else:
            print(f"SSH connection to {nodes} failed")


def configure_ray(list_of_nodes):

    user = os.path.expanduser('~')
    git_repo = "job_simulator"

    head_node, *worker_nodes = list_of_nodes[:]

    # clone important stuff at head




    if "job_simulator" not in os.listdir(user):
    # os.system(f"sudo rm -r {user}/{git_repo}")
        os.system(f"git clone git@gitlab.cs.tufts.edu:abdullah/{git_repo}.git {user}/{git_repo}")

    def do_in_parallel(nodeIP, user):
        os.system(f"ssh {nodeIP} sudo rm -r {user}/{git_repo}")        
        os.system(f"ssh {nodeIP} git clone 10.1.1.2:{user}/{git_repo}")

    threads=list()
    for nodeIP in worker_nodes:
        t = Thread(target=do_in_parallel, args=(nodeIP,user,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


    print("configuration done")


def rsync_cluster():
    for node in worker_nodes:
        os.system(f"rsync -av {user}/PCS {node}:{user}/")


def installer(exclude_head=False):


    if exclude_head:
        list_of_nodes = worker_nodes

    threads = list()
    for node in list_of_nodes:
        

        t = Thread(target=os.system, args=(f"ssh {node} <installer.sh",))
        t.start()
        threads.append(t)


    for t in threads:
        t.join()


    with open("/tmp/state", "w") as f:
        f.write("dependencies_installed")
    print("installation done!")



def cluster_reboot():
    print(f"Rebooting worker nodes")
    for node in worker_nodes:
        os.system(f"ssh {node} sudo reboot")
    print(f"Rebooting head node")
    os.system(f"sudo reboot")


def increase_file_limit():
    def check_ulimit(ip):
        command = f"ssh {ip} 'ulimit -n'"
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0 and result.stdout.strip() == "100000"


    limit = 100000

    with open("/etc/security/limits.conf", "r") as f:
        contents = f.readlines()

    values = [f"{user_name}            hard    nofile            100000\n",
            f"{user_name}            soft    nofile            100000\n",
            "root            hard    nofile            100000\n",
            "root            soft    nofile            100000\n"]

    for value in values:
        contents.insert(-4, value)


    with open("/tmp/limits.conf", 'w') as f:
        contents = "".join(contents)
        f.write(contents)

    os.system(f"sudo mv /tmp/limits.conf /etc/security/limits.conf")


    for node in worker_nodes:
        os.system(f"sudo rsync -e 'ssh -o StrictHostKeyChecking=no' /etc/security/limits.conf {node}:/etc/security/")


    for node in list_of_nodes:

        if check_ulimit(node):
            print(f"successfully set ulimit -n to 100000 on {node}")
        else:
            print(f"ulimit -n is NOT 100000 on {node}")

def setup_ray_cluster():
    import ray

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
        

    head_port = 6379
    ray_dir = get_ray_path()


    if not ray_dir:
        print("Ray not found")
        return

    # stopping all previous instances
    for node in list_of_nodes:
        os.system(f"ssh {node} {ray_dir} stop")

    # starting on head node
    os.system(f"{ray_dir} start --head --node-ip-address={head_node} --port={head_port} --redis-password=tf_cluster_123")


    for node in worker_nodes:
        time.sleep(1)
        os.system(f"ssh {node} {ray_dir} start --address={head_node}:{head_port} --redis-password=tf_cluster_123")

    time.sleep(5)

    ray.init(address="%s:%s" % (head_node, head_port), _redis_password="tf_cluster_123")

    ray_nodes = list(filter(lambda n: n["alive"], ray.nodes()))
    print("Num of nodes: %d" % len(ray_nodes))
    print(ray.cluster_resources())
    print(ray.available_resources())



if __name__ == '__main__':

    if os.path.exists("/tmp/state"):
        state = "dependencies_installed"
    else:
        state = None


    if state != "dependencies_installed":
        setup_keys()
        rsync_cluster()
        installer()
        cluster_reboot()

    setup_ray_cluster()
    increase_file_limit()