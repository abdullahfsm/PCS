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



if __name__ == '__main__':
    os.system(f'bash setup_keys.sh {N} {user_name}')

    
    # if len(sys.argv) != 2:
    #     start()
    # else:
    #     globals()[sys.argv[1]]()