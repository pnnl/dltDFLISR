#!/bin/bash
# chmod +x enviroment.sh --- to be able to run script
# ########################################################################## Start docker variables
pathTOgridappsdDocker="$HOME/grid/gridappsd-docker"
TAGgridappsD="v2024.06.0"; # tags: original "develop"
# ########################################################################## End docker variables
# Below this point it should not be altered

# Deleting and remaking the docker
echo "Going to Docker path: $pathTOgridappsdDocker."
cd $pathTOgridappsdDocker
./stop.sh -c
wait
if [ -d "./gridappsd" ]; then
    sudo rm -r gridappsd
    echo "directory gridappsd removed"
else
    echo "directory gridappsd does not exits"
fi
# docker rmi $(docker images -q)
docker rmi gridappsd-docker
wait
docker volume rm $(docker volume ls -q)
wait
# make docker image
bash run.sh -t $TAGgridappsD