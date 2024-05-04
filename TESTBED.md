### Follow these instructions for setting up the GPU testbed on cloudlab


## Starting a cloudlab experiment
Assuming that you have a cloudlab profile:
* Navigate to Create Experiment Profile [https://www.cloudlab.us/manage_profile.php]
* Upload the provided cloudlab profile (cloudlab_profile.py)
* Once the experiment profile has been set up, navigate to Start Experiment [https://www.cloudlab.us/instantiate.php] and choose the profile just created
* Starting the experiment may take some time
* Once the experiment is up and running you can ssh into the head node (n0)


## Setting up cluster
Assuming you can ssh into the head node of a cloudlab cluster:
* Run `git clone https://github.com/abdullahfsm/PCS.git` at root directory followed by `cd ~/PCS`, `git checkout osdi2024-artifacts` and finally `cd utils`
* Run `python3 cluster_utils.py install` This script takes a considerable time (~20 min) to install the required dependencies and upon successfully completing will reboot all the nodes in the cluster (you will have to login to the cloudlab head node again)
* Once all the cluster machines have been rebooted:
* * Login to the cloudlab head node again, navigate to `cd ~/PCS/utils`
* * Run `python3 cluster_utils.py launch` This will set up a ray cluster!
* * To verify that the cluster has successfully launched, run `ray status` which will show the resources and nodes available to the ray cluster. We are now ready to run our workloads!

## HelloWorld example
* Run `bash run_toy_testbed_example.sh` 