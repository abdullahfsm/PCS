sudo su fdisk /dev/sda
p
d
n
a
w

remove line from /etc/fstab
reboot

resize2fs /dev/sda1

Need to setup ssh keys between nodes to do ssh 10.1.1.2
Need to setup ssh key between git and node (n0)
git clone 10.1.1.2:/users/abdffsm/name_of_gitrepo
viola



git clone https://github.com/ray-project/ray.git
git remote remove origin
git remote add origin git@gitlab.cs.tufts.edu:abdullah/ray.git
viola
