### Follow these instructions for setting up the GPU testbed on cloudlab


## Staring a cloudlab experiment
Assuming that you have a cloudlab profile:
* Navigate to Create Experiment Profile [https://www.cloudlab.us/manage_profile.php]
* Upload the provided cloudlab profile (cloudlab_profile.py)
* Once the experiment profile has been set up, navigate to Start Experiment [https://www.cloudlab.us/instantiate.php] and choose the profile just created
* Starting the experiment may take some time
* Once the experiment is up and running you can ssh into the head node (n0)


## Setting up cluster
Assuming you can ssh into the head node of a cloudlab cluster:
* Run `git clone https://github.com/abdullahfsm/PCS.git` at root directory followed by `cd ~/PCS/utils`
* Run `python3 cluster_utils.py install` This script takes a considerable time (~15 min) to install the required dependencies and upon successfully completing will reboot all the nodes in the cluster(you will have to login to the cloudlab head node again)
* Once all the cluster machines have been rebooted:
* * login to the cloudlab head node again, navigate to `cd ~/PCS/utils`
* * run `python3 cluster_utils.py launch` This will set up a ray cluster you should be able to see the number of nodes, total cpu and gpu resources available to the ray cluster. We are now ready to run our workloads!