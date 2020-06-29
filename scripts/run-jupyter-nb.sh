#!/bin/bash

echo "[INFO]"
echo "[INFO] ## ============================================== ##"
echo "[INFO] ## $(basename ${BASH_SOURCE[0]})"
echo "[INFO] ## ============================================== ##"
echo "[INFO]"

# Warning:
# In order to access the JupyterLab from other machine on the network,
# please be sure to run the container with "docker run --network="host" ..."


#
# globals
#

# ip="$(ip -f inet address show eth0 | grep -Po 'inet \K[\d.]+')"
ip="$(hostname -i)"
port=8080
top_dir=${1}

#
# usage()
#

usage() {

	echo "[INFO]"
	echo "[INFO] ./${0} TOP_DIR"
	echo "[INFO]"

	return 0

}

#
# main
#

if [ ! -d "${top_dir}" ]; then

	echo "[ERROR] Directory ${top_dir} is not defined. Abort!"
	usage
	exit -1

fi

echo "[INFO] Running Jupyter notebook on IP=${ip} and port=${port}:"

cd $top_dir
jupyter notebook --ip=${ip} --port=${port} --no-browser


