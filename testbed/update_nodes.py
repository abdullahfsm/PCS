import os, sys
from threading import Thread



# remove ray_results, git add, git commit, git push on auto_ml_dir, ray_dir

def git_push_update_fused(ip, path):
    os.system("ssh %s git -C %s add %s/." % (ip, path, path))
    os.system("ssh %s git -C %s commit -m latest_auto" % (ip, path))
    os.system("ssh %s git -C %s push -u" % (ip, path))
    
def git_pull_update_fused(ip, path):

    os.system("ssh %s git -C %s reset --hard HEAD" % (ip, path))
    os.system("ssh %s git -C %s pull --recurse-submodules" % (ip, path))
    os.system(f"ssh {ip} git -C {path} submodule update --init --recursive --remote")


def pull_worker(worker_ip, auto_ml_dir, ray_dir, ray_results_dir):
        os.system(f"ssh {worker_ip} sudo rm -r {auto_ml_dir}/__pycache__/")
        git_pull_update_fused(worker_ip, auto_ml_dir)
        git_pull_update_fused(worker_ip, ray_dir)
        os.system(f"ssh {worker_ip} rm -r {ray_results_dir}")

if __name__ == '__main__':


    with open("/var/emulab/boot/hostmap") as fp:
        N, *_ = fp.readlines()
    N = int(N.rstrip())


    head_ip, *worker_ips = ["10.1.1.%d" % (2+node_id) for node_id in list(range(N))]

    auto_ml_dir = "/users/abdffsm/automl-setup"
    ray_dir = "/users/abdffsm/ray"
    ray_results_dir = "/users/abdffsm/ray_results"
    
    # on head ip
    os.system(f"sudo rm -r __pycache__/")
    os.system(f"ssh {head_ip} git -C {auto_ml_dir} submodule update --init --recursive --remote")
    git_push_update_fused(head_ip, auto_ml_dir)
    git_push_update_fused(head_ip, ray_dir)
    os.system(f"sudo rm -r {ray_results_dir}")    


    threads=list()
    for worker_ip in worker_ips:

        t = Thread(target=pull_worker, args=(worker_ip,auto_ml_dir,ray_dir,ray_results_dir,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("updated")