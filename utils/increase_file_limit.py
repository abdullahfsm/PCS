import os
import sys
import fileinput

if __name__ == '__main__':
	
	limit = 100000

	with open("/etc/security/limits.conf", "r") as f:
	    contents = f.readlines()




	values = ["abdffsm            hard    nofile            100000\n",
	"abdffsm            soft    nofile            100000\n",
	"root            hard    nofile            100000\n",
	"root            soft    nofile            100000\n"]


	for value in values:
		contents.insert(-4, value)

	with open("/etc/security/limits.conf", "w") as f:
	    contents = "".join(contents)
	    f.write(contents)

	with open("/var/emulab/boot/hostmap") as fp:
	    N, *_ = fp.readlines()
	N = int(N.rstrip())
	head_ip, *worker_ips = ["10.1.1.%d" % (2+node_id) for node_id in list(range(N))]

	for ip in worker_ips:
		os.system(f"rsync /etc/security/limits.conf {ip}:/etc/security/")

