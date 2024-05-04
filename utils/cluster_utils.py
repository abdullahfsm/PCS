import os
import sys
from threading import Thread
import time
import subprocess
import fileinput

user = os.path.expanduser('~')
user_name = user.split('/')[-1]


with open("/var/emulab/boot/hostmap") as fp:
    N, *_ = fp.readlines()
N = int(N.rstrip())
list_of_nodes = ["10.1.1.%d" % (2+node_id) for node_id in list(range(N))]

head_node, *worker_nodes = list_of_nodes[:]
head_port = 6379


def setup_keys():
    def check_ssh(ip):


        command = f"ssh -o BatchMode=yes {ip} exist"

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()

        return process.returncode == 0


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
    def check_ulimit(ip, limit):
        command = f"ssh {ip} 'ulimit -n'"

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()


        return process.returncode == 0 and stdout.strip() == limit

    limit = 100000

    with open("/etc/security/limits.conf", "r") as f:
        contents = f.readlines()

    values = [f"{user_name}            hard    nofile            {limit}\n",
            f"{user_name}            soft    nofile            {limit}\n",
            f"root            hard    nofile            {limit}\n",
            f"root            soft    nofile            {limit}\n"]

    for value in values:
        contents.insert(-4, value)


    with open("/tmp/limits.conf", 'w') as f:
        contents = "".join(contents)
        f.write(contents)

    os.system(f"sudo mv /tmp/limits.conf /etc/security/limits.conf")


    for node in worker_nodes:
        os.system(f"sudo rsync -e 'ssh -o StrictHostKeyChecking=no' /etc/security/limits.conf {node}:/etc/security/")


    for node in list_of_nodes:

        if check_ulimit(node, limit):
            print(f"successfully set ulimit -n to {limit} on {node}")
        else:
            print(f"ulimit -n is NOT {limit} on {node}")



def install():
    setup_keys()
    rsync_cluster()
    installer()
    increase_file_limit()
    cluster_reboot()



def get_ray_path():
    # Run the command and capture its output

    command = f"which ray"

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        return stdout.strip()
    return None


def tear_down():
    import ray

        
    ray_dir = get_ray_path()


    if not ray_dir:
        print("Ray not found")
        return

    # stopping all previous instances
    for node in list_of_nodes:
        os.system(f"ssh {node} {ray_dir} stop")



def launch():
    import ray
        

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


def get_status():
    import ray
    ray.init(address="%s:%s" % (head_node, head_port), _redis_password="tf_cluster_123", conda="osdi24")

    ray_nodes = list(filter(lambda n: n["alive"], ray.nodes()))
    print("Num of nodes: %d" % len(ray_nodes))
    print(ray.cluster_resources())
    print(ray.available_resources())




if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f"usage: python3 cluster_utils.py install/launch")
    else:

        try:
            val = globals()[sys.argv[1]]()
            if val:
                print(val)
        except Exception as e:
            raise e