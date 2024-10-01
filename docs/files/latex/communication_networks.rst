
Simulation of Communication Networks in a Co-Simulation Environment
====================================================================

Introduction
-------------------------------------------------------------------

Co-simulation is a computational technique where the global simulation of a system of coupled, heterogeneous components is done by composing local simulations of the individual components.
Each simulation unit encapsulates the behavior and characteristics of a specific component and may be developed independently by different software teams.
The simulation units can be thought of independently as “black boxes.”
Moreover, the simulation units themselves may be executed on different machines, operating systems, and networks.
The coupling is performed by an orchestrator that initializes each simulation unit, controls how simulated time progresses for each simulation unit, and manages communication and the transfer of data between simulation units according to the co-simulation scenario that is being computed.

The key advantage of co-simulation is that it enables the global simulation of arbitrarily complex, designed, but not yet built, systems by composing preexisting local simulations in novel ways.
A comprehensive survey of co-simulation approaches that have been applied to various domains of discourse is given by :cite:`gomes2018co`.



Simulation of Communication Networks
-------------------------------------------------------------------

The future implementation of smart power grids will rely upon the successful development and large-scale deployment of heterogeneous, connected devices, including appliances and meters at the customer’s site and sensors in the transmission and distribution grids, enabling networked control and demand response.
Co-simulation approaches can be used to evaluate the efficacy of these networked devices, including their responsiveness to market conditions and dynamics such as instantaneous energy cost, and the robustness of the communication networks themselves in the presence of lossy communication channels, faults, and other quality-of-service disruptions.

Power system dynamic simulation is typically done as a continuous time simulation where the communication networks are modeled as discrete event systems that account for the randomness of packet generation and transmission.
This approach was pioneered by :cite:`hopkinson2003distributed` with their contribution of Electrical Power and Communication Synchronizing Simulator (EPOCHS).
Later proposals by other groups contributed in 3 research directions: exploration of different combinations of power system and communication network simulation units, improvement of time synchronization mechanisms, and improvement and standardization of software engineering techniques and programming frameworks.

In continuous time systems, the state variables for the system vary continuously with respect to time.
The systems may be represented as sets of coupled differential equations that codify the relationships between the state variables and their rates of change.
In the simplest cases, the differential equations may be solved analytically, obtaining closed form solutions.
However, for most real-world cases closed forms tend to not be available, and instead, numerical solutions may be obtained by discretizing the differential equations and integrating over small changes to the state variables, approximating the system’s trajectory.

In discrete event systems, the state variables for the system are subject to change due to discrete events whose occurrences are irregular with respect to time.
Hence, time discretization into regular intervals, as would be done for a continuous system, is not possible, as an appropriate time step cannot always be determined *a priori*.
If the time step is too large, then some events may not be observed, whereas if the time step is too small, then computational performance will suffer because there will be many time steps with no events.
Instead, in event-driven simulations, a scheduler manages the current simulation time and maintains a list of events.
Simulation units may add events to the list, which are then managed by the scheduler.
The global simulation concludes when the simulation time reaches a predetermined stopping time and/or when the system enters a predetermined state.

In :cite:`liberatore2011smart`, the authors propose PowerNet, a communication network for smart power grids, and evaluate it with respect to network metrics, including measured delay and jitter, control metrics, including stability, and operational metrics, including scalability, security, safety, and performance.
The communication network is simulated at the packet-level using the ns-2 discrete event network simulator :cite:`breslau2000advances` and the power grid and connected devices are simulated using Modelica :cite:`fritzson1998modelica`.
Modelica and ns-2 are executed as 2 separate processes, managed and scheduled by the operating system, and the communication between processes is done via UNIX pipes that are orchestrated by ns-2.
A key disadvantage of this approach to time synchronization is that ns-2 effectively determines when communication can occur between the simulation units, precluding the ability to simulate control and/or alarm signals that emanate from the connected devices.

In :cite:`ciraci2014fncs`, the authors propose FNCS, which utilizes GridLAB-D :cite:`chassin2008gridlab`, PowerFlow (developed as a component of FNCS), and ns-3 :cite:`riley2010ns`.
To mitigate is overhead of time synchronization between simulation units, FNCS uses a speculative execution that attempts to predict when simulation units are going to exchange data.
A similar approach for power system and network communication co-simulation is advanced by :cite:`lin2011power`, which describes a combination of ns-2 and Positive Sequence Load Flow (PSLF) software package by GE.
However, as with the previously described approaches, the overhead of time synchronization remains a prevailing issue.

More recently, in :cite:`shum2018co`, the authors present the DecompositionJ framework that demonstrates how source code compiler extensions can be used to automatically transform multi-threaded co-simulations into direct-execution simulations where the original source code for the target program is modified to emulate its own behavior.
The key advantage of this approach is that it enables the system to model and account for computation delays, which may result in timing and synchronization issues.
Control code is automatically injected into the original source code to manage the target program’s interactions with real-world systems such as I/O and the machine’s clock, to account for computation delays, and to synchronize events across simulation units.

Considerations for future research and development of smart power grid co-simulations are articulated in :cite:`steinbrink2018future`, where the authors identify the need for combining the advantages of hybrid coupling between simulation units and programming framework usability, e.g., standardized application programming interfaces, automatic validation of simulation scenarios, and the development of graphical user interfaces).



Conclusion
-------------------------------------------------------------------

This article introduces the co-simulation computational technique and discusses how it has been applied to model the behavior of power grids using a combination of continuous time simulations and discrete event systems.
The core research and development challenges and opportunities are to identify suitable software packages for the various simulation units, reducing the overhead of time synchronization between simulation units, and improving and standardizing the available software engineering techniques and programming frameworks.

.. bibliography:: document.bib
   :style: unsrt

