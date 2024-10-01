## Distributed Fault Location, Isolation, and Service Restoration (DFLISR)

This repository contains the Distributed Fault Location, Isolation, and Service Restoration (DFLISR) application for power distribution systems. One approach to minimizing customer impact after a permanent fault is use of a fault location, isolation, and service restoration (FLISR) algorithm that reconfigures topology through a centralized controller. Permanent faults cannot be cleared by tripping and reclosing, and without a recloser, all faults are considered permanent. An autonomous, DFLISR solution would improve grid resilience. The application is focus on the data separation of switch delimited areas. More information can be found in the paper [doi:10.1109/ACCESS.2023.3287831](https://ieeexplore.ieee.org/document/10159249). The mentioned paper explores the potential of distributed ledger technology (DLT) to improve fault-tolerant grid operations. The approach is not limited to DFLISR. However, this repository is focussed exclusively on the DFLISR. 

Currently a single agent is populated for every switch delimited area. The agent has limited information of the surrounding switch delimited area. 

## Prerequisites

The proposed application is developed in Python, and it requires multiple packages to be able to execute the application. Note that all these packages can be installed using pip. The core of the approach leverages GridAPPS-D and Python Poetry to create the python environment capable of running the application.

### GridAPPS-D

GridAPPS-D can be install and configure to operate with the current application in multiple ways. The method utilize by this application follows the Windows 10 WSL guide.

[GridAPPS-D Documentation](https://gridappsd-training.readthedocs.io/en/develop/#)

By following the documentation you will have GridAPPS-D install in a docker container. Once the docker containers are installed make sure they are running and attach to the GridAPPS-D container.

The shell script below starts the GridAPPS-D from withing the docker. To confirm, open localhost:8080 to access the GridAPPS-D Visualization App on your browser.

```shell
./run-gridappsd.sh
```

### Python Environment

The python environment utilized for running the application utilizes **poetry**. Python Poetry is a dependency management and packaging tool for Python. It aims to provide an easy and reliable way to manage project dependencies, build projects, and publish them. Alternatively a python environment that satisfy the applications requirements can be utilized.

To create the Python Poetry application environment do as shown in the shell script below.

```shell
poetry install
poetry update
```

## Running the Application

### Configuring the Power Distribution Feeder
The GridAPPS-D leveraged is the distributed app architecture presented in [doi:10.1109/ACCESS.2024.3374331](https://ieeexplore.ieee.org/document/10462129). In order to execute the application the GridAPPS-D container must be running a matching power distribution feeder. The setting of the GridAPPS-D matching power distribution feeder can be performed by running the script below:

```shell
docker cp config/pnnl.goss.gridappsd.cfg  gridappsd:/gridappsd/conf/pnnl.goss.gridappsd.cfg
```

### Running the Application
run main.py from the terminal in the main directory.

```shell
poetry run python src/main.py
```

## Application Overview UML
The user by running the application will create the Feeder Agent. The Feeder Agent will populate the Switch Area Agents. The Agents will interact with each other and GridAPPS-D by a single Data Bus. GridAPPS-D is performing the power distribution system network simulation reporting measurements and receiving commands to open/close circuit breakers to perform the DFLIR system changes for reconfiguration.

```{uml}
actor User

participant "Feeder Agent" as Feeder
participant "Data Bus" as Bus

participant "Switch Area Agent 1" as Agent1
participant "Switch Area Agent 2" as Agent2
participant "Switch Area Agent 3" as Agent3
participant "Switch Area Agent 4" as Agent4
participant "Switch Area Agent 5" as Agent5

participant "GridAPPS-D" as GridAPPS_D

User -> Feeder: Initiates Feeder
Feeder -> Agent1: Creates Agent 1
Feeder -> Agent2: Creates Agent 2
Feeder -> Agent3: Creates Agent 3
Feeder -> Agent4: Creates Agent 4
Feeder -> Agent5: Creates Agent 5

GridAPPS_D -> Bus: Sends Measurements

Bus -> Feeder: Forwards Measurements
Bus -> Agent1: Forwards Measurements
Bus -> Agent2: Forwards Measurements
Bus -> Agent3: Forwards Measurements
Bus -> Agent4: Forwards Measurements
Bus -> Agent5: Forwards Measurements

Agent1 -> Bus: Sends Message to Agent 5
Bus -> Feeder: Forwards Message
Feeder -> Feeder: Adds Message Delay and pacage losses
Feeder -> Bus: Forwards Message to Agent 5
Bus -> Agent5: Sends Message from Agent 1

```

>**Note**:<br>
> Making updates to the documentation from the WSL python environment with will require having graphviz install in wsl. To perform the installation run ```sudo apt-get install graphviz```



