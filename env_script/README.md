## Scripts to create the environment and assist in configuration and debugging  

The scripts for creating and populating the environment require the folder organization described below. 

```{uml}
' skinparam defaultFontSize 16

folder $HOME/grid {
    folder gridappsd-docker as F1 { 
        [https://github.com/GRIDAPPSD/gridappsd-docker.git]
    }
    folder dltdflisr as F2 {
        [https://github.com/pnnl/dltDFLISR.git]
    }
    folder CIMHub as F3 { 
        [https://github.com/GRIDAPPSD/CIMHub.git]
    }
    folder Powergrid-Models as F4 { 
      [https://github.com/GRIDAPPSD/Powergrid-Models.git] 
      }
}

F1 -d[hidden]-> F3
' F2 -d[hidden]-> F3
F2 -d[hidden]-> F4
```

### environment.sh
* Stops GridAPPS-D docker.
* Deletes GridAPPS-D docker and all images.
* Deletes all docker volumes. **NOTE:** Deletes all not just GridAPPS-D docker volumes.
* Create GridAPPS-D docker.

### adding_test_case.sh
* Populates the test case on the docker.
* Places the needed GridAPPS-D configuration file ".cfg" in GridAPPS-D.

### show_test_cases.sh
* Shows the available tests cases in GridAPPS-D.

**NOTE:** Not needed for environment or run.

### port_utilization.sh
* Presents the port utilization.
* The default port is the 61613 used by GridAPPS-D. But it can be easily altered.
  * Can provide useful information for debuting.

**NOTE:** Not needed for environment or run.
