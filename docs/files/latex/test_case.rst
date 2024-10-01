.. _`sec:testsystem`:



Test System
===========


.. _fig_system:

.. figure:: testsystem.png
    :scale: 15 %
    :align: center

    One line diagram of the test system. Adapted from :cite:`fernando2021iowa_testcase`.

To evaluate the proposed methodology, we applied it to a distribution network to test its accuracy. Specifically, we utilized the publicly available Midwest 240-Node test distribution system, which is modeled after an actual distribution network located in the Midwest region of the United States :cite:`bu2019timeWEB, Wang_NAPS_2019`. This system consists of 240 primary network nodes and spans 23 miles of primary feeder lines. A visual depiction of the system can be found in Figure :ref:`fig_system`. We also leveraged a full year's worth of smart meter measurements at the node level. Appliance-level load data with minute-level resolution was synthesized for this period using the approach detailed in :cite:`FBR2020synthetic`, with the original data derived from hourly nodal smart meter readings. The synthesized load data achieved a mean absolute percentage error of 2.58% when compared to the hourly smart meter data. Both the minute-resolution load data and the GridLAB-D :cite:`GLD_PNNL_2022` model of the Midwest 240-Node test system are publicly available :cite:`FBR_Gridlabd_queue`. 

The Midwest 240-Node test distribution system is a radial network composed of three feeders—S, M, and L—denoting small, medium, and large sizes. Its accessibility, the availability of load profiles from real smart meter data, and the inclusion of multiple feeders with switch-delimited areas make it ideal for testing the proposed algorithm. Additionally, the system links each load bus to its respective households, which makes it well-suited for assessing the DLT-based FLISR’s effectiveness in restoring customer connections. Due to the state estimation (SE) algorithm's inability to handle split-phase transformers, the loads were assigned to the primary side of these transformers. A SE algorithm and a power network model reduction algorithm have been utilized but are not the focus.

.. bibliography:: references.bib
   :style: unsrt
