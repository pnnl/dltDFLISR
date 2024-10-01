#!/bin/bash
# Show available test cases
# ########################################################################## Start Show available test cases variables
listFEEDERSpath="$HOME/grid/Powergrid-Models/platform/"
listFEEDERS="list_feeders.sh"
# ########################################################################## End Show available test cases variables
# Below this point it should not be altered
get_test_cases () {
    pushd $listFEEDERSpath
    source $listFEEDERS
    wait
    popd
}
echo ""
get_test_cases
# echo $PWD
