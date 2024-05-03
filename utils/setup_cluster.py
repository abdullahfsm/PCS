import os, sys
from threading import Thread


user = os.path.expanduser('~')
with open("/var/emulab/boot/hostmap") as fp:
    N, *_ = fp.readlines()
N = int(N.rstrip())
list_of_nodes = ["10.1.1.%d" % (2+node_id) for node_id in list(range(N))]

head_node, *worker_nodes = list_of_nodes[:]



def installer(list_of_nodes):

    threads = list()
    for nodeIP in list_of_nodes:
        

        t = Thread(target=os.system, args=(f"ssh {nodeIP} <installer.sh",))
        t.start()
        threads.append(t)


    for t in threads:
        t.join()

    print("installation done!")


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

def create_partition(list_of_nodes):
    
    head_node, *worker_nodes = list_of_nodes[:]

    for nodeIP in worker_nodes:
        os.system(f"ssh {nodeIP} <create_partitions.sh")

    os.system(f"ssh {head_node} <create_partitions.sh")


def rsync_cluster(worker_nodes):
    for node in worker_nodes:
        os.system(f"rsync -av {user}/PCS {node}:{user}/")


if __name__ == '__main__':
    rsync_cluster


    # for num in {2..21}; do scp jmetal_ray-1-py3-none-any.whl "10.1.1.$num:/users/abdffsm"; done

    # create_partition(list_of_nodes)
    # installer(list_of_nodes)
    # configure_ray(list_of_nodes)

