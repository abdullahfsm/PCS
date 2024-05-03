"""N-node-GPU Cluster

Instructions:
Specify number and type of machines.
"""


#
# NOTE: This code was machine converted. An actual human would not
#       write code like this!
#

# Import the Portal object.
import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as pg
# Import the Emulab specific extensions.
import geni.rspec.emulab as emulab

# Create a portal object,
pc = portal.Context()


# getting user input on number of nodes
pc.defineParameter( "n", "Number of nodes", portal.ParameterType.INTEGER, 4)
pc.defineParameter( "ntype", "type of nodes", portal.ParameterType.STRING, 'c240g5')

params = pc.bindParameters()


# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()

ifaces = list()


# Creating a lan
lan = request.Link()
lan.bandwidth = 20000000

for node in range(params.n):
    node_n0 = request.RawPC('n%d'%node)
    node_n0.hardware_type = params.ntype
    #node_n0.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD'
    node_n0.disk_image = 'urn:publicid:IDN+clemson.cloudlab.us+image+tuftscc-PG0:cuda-files-ubuntu18'
    iface0 = node_n0.addInterface('interface-1%d'%node, pg.IPv4Address('10.1.1.%d'%(2+node),'255.255.255.0'))
    lan.addInterface(iface0)
    

# Print the generated rspec
pc.printRequestRSpec(request)