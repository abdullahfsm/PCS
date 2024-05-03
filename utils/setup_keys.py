import os
import time
import subprocess
import sys


user = os.path.expanduser('~')

with open("/var/emulab/boot/hostmap") as fp:
    N, *_ = fp.readlines()
N = int(N.rstrip())
head_ip, *worker_ips = ["10.1.1.%d" % (2+node_id) for node_id in list(range(N))]





if __name__ == '__main__':
    print(user)
    print(head_ip)
    print(worker_ips)


    
    # if len(sys.argv) != 2:
    #     start()
    # else:
    #     globals()[sys.argv[1]]()