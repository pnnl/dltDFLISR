import time
import logging
import copy
import os
import sys
import pandas as pd
from datetime import datetime, timezone, timedelta
# GridAPPS-D
from sim import Sim
import agents as AG
from cimgraph.data_profile import CIM_PROFILE
import gridappsd.field_interface.agents.agents as agents_mod

from gridappsd.difference_builder import DifferenceBuilder
from gridappsd.field_interface.interfaces import MessageBusDefinition
import gridappsd.topics as t


from gridappsd.field_interface.context import LocalContext
# Helper functions
import helper as HP
import area_helper_functions as PSA


#### Initialize log and file location
log = logging.getLogger(__name__)
file_location = os.path.dirname(os.path.abspath(__file__))

#### Required variables
cim_profile = CIM_PROFILE.RC4_2021.value
agents_mod.set_cim_profile(cim_profile, iec61970_301=7)

#### constant variables
class InitializationHardCodedTimeSleep:
    """A hard-coded configuration class with number variables.

    Provide the sleep times needed to properly initialize the simulation

    Attributes:
        interval_1 (float): Time interval in seconds. Await simulation to start.
        interval_2 (float): Time interval in seconds. Internal loop to pause the simulation.
        interval_3 (float): Time interval in seconds. Await first pause stop to be concluded.
        interval_4 (float): Time interval in seconds. Running second at a time and waiting to get second precision simulation time.
        interval_5 (float): Time interval in seconds. Await CB changes to take full effect.
        interval_6 (float): Time interval in seconds. Internal loop for initializing the agents.
        interval_7 (float): Time interval in seconds. Time wait before emulating the fault event.
        interval_8 (float): Time interval in seconds. Needed time for the emulating fault event to take effect.
        internal_9 (float): Time interval in seconds. Internal loop verify if the simulation is done.
    """
    interval_1 = 15.0 # time.sleep(15) await simulation to start
    interval_2 = 0.1 # time.sleep(0.1) internal loop pausing the simulation
    interval_3 = 30.0 # time.sleep(30) await first pause stop to be concluded 
    interval_4 = 3.0 # time.sleep(3) running second at a time and waiting to get second precision simulation time
    interval_5 = 30.0 # time.sleep(10) await CB changes to take full effect  
    interval_6 = 15.0 # time.sleep(10) internal loop for initializing the agents 
    interval_7 = 3.0 # time await before emulating the fault event 
    interval_8 = 1.0 # needed time for the emulating fault event to take effect
    interval_9 = 0.2 # Internal loop verify if the simulation is done.

#### Functions
def main(configuration: dict) -> None:
    """Main entry point for the simulation.

    This script initializes the simulation environment, loads required data, starts the simulation, populate the agents, and once the simulation is done the outputs are written to file.
    The results are saved to the specified output directory as configure. The argument configuration contains all the required information.
    This script can be run from the command line the **if __name__ == "__main__":** portion contain the file being loaded with the configuration.

    Example:
        Run the simulation with example settings:
        
            $ python /src/main.py

    Args:
        configuration (dict): Contains all the required information. Please refer to the larger documentation for details.
    """
    global feeder_agent
    global area_agents
    #### additional data read
    ybus=PSA.additional_area_data(configuration)
    #### begin simulation
    log.debug("make and start simulation")
    sim = Sim(configuration['test_case'])
    time.sleep(InitializationHardCodedTimeSleep.interval_1)
    for i in range(100):
        sim.simulation.pause()
        time.sleep(InitializationHardCodedTimeSleep.interval_2)
    #### populate agents
        #### populate feeder agent
    system_bus = HP.overwrite_parameters(sim.get_feeder_id())
    feeder_bus = system_bus
    log.debug("Making AG.SampleFeederAgent.")
    feeder_agent = AG.SampleFeederAgent(feeder_bus, system_bus, configuration['app_config'], None, sim.get_simulation_id(),sim,configuration)
        #### having time information in feeder agent
    log.debug(f"Providing feeder the simulation time")
    sim.simulation.resume_pause_at(60)
    time.sleep(InitializationHardCodedTimeSleep.interval_3)
    for _ in range(65):
        feeder_agent.time_control()
        time.sleep(InitializationHardCodedTimeSleep.interval_4)
    log.debug("Making AG.SampleSwitchAreaAgent.")
    area_agents=list()
    switch_areas = feeder_agent.agent_area_dict['switch_areas']
        #### CB initial condition
    log.debug(f"Setting CB to initial condition.")
    cb=configuration['feeder_agent_configuration']['initial_CB_state'].split("_")[-1]

    temp=feeder_agent.system_status
    feeder_agent.system_status = feeder_agent.system_status.replace("1", "5").replace("0", "5")
    feeder_agent.update_cb_status(cb)
    feeder_agent.system_status = temp
 
    time.sleep(InitializationHardCodedTimeSleep.interval_5) 
        #### populate switch agents
    for sw_idx, switch_area in enumerate(switch_areas):
        name=""
        for key, value in configuration["feeder_agent_configuration"]["switch_area_names"].items():
            if set(switch_area["boundary_switches"]) == set(value):
                name = key
                configuration['area_agent_configuration']['ybus']=ybus[name]
                break    
        switch_bus = HP.overwrite_parameters(sim.get_feeder_id(), f"{sw_idx}")
        # additional exclusive area information (i.e., ybus)
        area_agents.append( AG.SampleSwitchAreaAgent(feeder_bus, switch_bus, configuration['app_config'], switch_area, sim.get_simulation_id(), name=name, agent_config=copy.deepcopy(configuration)) )
        log.debug(f"Created agent: {sw_idx}")
    #### initializing the agents
    for _ in range(65):
        feeder_agent.time_control()
        time.sleep(InitializationHardCodedTimeSleep.interval_6)
    #### start feeder agent time control
    log.debug(f"Starting feeder at time control")
    feeder_agent._perform_self_time_control_call=True # feeder_agent._perform_self_time_control_call=False
    feeder_agent.time_control()
    #### Emulate fault
    time.sleep(InitializationHardCodedTimeSleep.interval_7)
    log.debug(f"Emulate fault")
    for i in configuration['feeder_agent_configuration']['forced_event']['breaker_status_change']:
        feeder_agent.open_CB(i)
        time.sleep(InitializationHardCodedTimeSleep.interval_8) # needed time to all commands to take effect 
    log.debug(f"SImulation is at fault.")
    #### awaiting simulation to end
    while not sim.done:
        try:
            time.sleep(InitializationHardCodedTimeSleep.interval_9)
        except (Exception, KeyboardInterrupt) as e:
            log.info(f"Simulation stop due to: {e}")
            break
    feeder_agent.monitoring_to_file() # save monitor data to file
            
if __name__ == "__main__":
    # read TOML      os.path.expanduser('~/grid/dltdflisr/config/input_config.toml')
    configuration = HP.load_toml(f'{file_location}/../config/input_config.toml')
    # configure logging
    HP.configure_logging(**configuration['logging_configuration'])
    # create sport environmental variables
    HP.os_variables(configuration['system_variables'])
    #### run main
    main(configuration)

    
