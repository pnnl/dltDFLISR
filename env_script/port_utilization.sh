#!/bin/bash
# chmod +x port_utilization.sh --- to be able to run script
# ########################################################################## Start docker variables
port="61613"
# ########################################################################## End docker variables
# Below this point it should not be altered

# Identify the process using the port .... 
echo "Identifying process using port $port..."
process_id=$(powershell.exe -Command "Get-Process -Id (Get-NetTCPConnection -LocalPort $port).OwningProcess")

if [[ -z "$process_id" ]]; then
    echo "No process is using port $port."
else
    echo "Process using port $port: $process_id"
fi


