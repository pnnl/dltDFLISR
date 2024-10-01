#!/bin/bash
# chmod +x adding_test_case.sh --- to be able to run script
# ########################################################################## Start adding teste case to gridappsd variables
listFEEDERSpath="$HOME/grid/Powergrid-Models/platform/"
listFEEDERS="list_feeders.sh"
fname="test240b_open_76591_CIM100x.XML"
caseNandMRID="test240b_open_76591 _EC8F84FC-AB8D-4C16-86CA-5F9382CD7892"
pathTOcaseCREATION="$HOME/grid/CIMHub"
# ########################################################################## End adding teste case to gridappsd variables
# Below this point it should not be altered
curent_path=$PWD
# Adding the case script
get_list_test_cases () {
    pushd $listFEEDERSpath
    source $listFEEDERS
    wait
    popd
}
echo ""
get_list_test_cases
echo "Start adding the test case"
echo ""

# Cleaning test case location
echo "Go to creation location."
cd $pathTOcaseCREATION
git clean -fdx

# ADD .XML tescase file to the creation location.
echo "Add .XML tescase file to the creation location."
cp "$curent_path/../config/$fname" "$pathTOcaseCREATION/$fname"


# Adding test case to GRIDAPPS-D
echo $fname
curl -s -D- -H 'Content-Type: application/xml' --upload-file $fname -X POST 'http://localhost:8889/bigdata/namespace/kb/sparql'
wait
# sleep 5
echo $caseNandMRID
python3 ./utils/ListMeasureables.py cimhubconfig.json $caseNandMRID
wait
# sleep 5
for f in `ls -1 *txt`; do python3 ./utils/InsertMeasurements.py cimhubconfig.json $f uuidfile.json; done
echo ""
echo "After adding the test case"
get_list_test_cases

# fix cfg file
docker cp "$HOME/grid/dltdflisr/config/pnnl.goss.gridappsd.cfg"  gridappsd:/gridappsd/conf/pnnl.goss.gridappsd.cfg

# reopens the gridappsd docker
# docker exec -it gridappsd bash