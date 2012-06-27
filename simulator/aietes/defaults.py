import numpy.random as rand
#####################################################################
# Simulation
#####################################################################
class Simulation():
    channel_event_name="AcousticEvent"
#####################################################################
# Nodes
#####################################################################
class Nodes():
    naming_convention={
        "Zeus",
        "Neptune",
        "Jupiter",
        "Hera",
        "Mercury",
        "Faunus",
        "Hestia",
        "Demeter",
        "Poseidon",
        "Diana",
        "Apollo",
        "Pluto",
        "Juno",
        "Vesta",
        "Diana",
        "Hermes",
        "Persephone",
        "Cupid",
        "Ares",
        "Hephaestus"
    }

#####################################################################
# Environment
#####################################################################
class Environment():
    shape=[100,100,100]
    scale=1
    base_depth=-1000

#####################################################################
# Application Layer
#####################################################################
class Application():
    pass
#####################################################################
# Network Layer
#####################################################################
class Network():
    pass
#####################################################################
# Media Access Layer
#####################################################################
class MAC():
    protocol="ALOHA"
    ack_packet_length=24
#####################################################################
# Physical Layer
#####################################################################
class PHY():
    frequency=20.0
    bandwidth=1.00
    bandwidth_to_bit_ratio=1.00
    variable_bandwidth=False
    transmit_power=250
    max_transmit_power=transmit_power
    recieve_power=-3
    listen_power=-30.00
    var_power={0:1419.73,
               1:2839.45,
               2:4259.18,
               3:5678.91}
    threshold={'SNR':20.0,
               'SIR':15.0,
               'LIS':3.0
              }

