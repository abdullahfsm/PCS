import os
import time
import subprocess
import sys


user_path = os.path.expanduser('~')
user_name = user_path.split('/')[-1]


with open("/var/emulab/boot/hostmap") as fp:
    N, *_ = fp.readlines()
N = int(N.rstrip())
head_ip, *worker_ips = ["10.1.1.%d" % (2+node_id) for node_id in list(range(N))]



def check_ssh(ip):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', ip, 'exit'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


if __name__ == '__main__':
    os.system(f'bash setup_keys.sh {N} {user_name}')
    time.sleep(1)

    ips_to_check = [head_ip] + worker_ips
    for ip in ips_to_check:
        if check_ssh(ip):
            print(f"SSH connection to {ip} successful")
        else:
            print(f"SSH connection to {ip} failed")


    
    # if len(sys.argv) != 2:
    #     start()
    # else:
    #     globals()[sys.argv[1]]()