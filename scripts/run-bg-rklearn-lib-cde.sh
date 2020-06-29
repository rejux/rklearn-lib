#!/bin/bash

echo -e "[INFO]"
echo -e "[INFO] ## ============================================== ##"
echo -e "[INFO] ## $(basename ${BASH_SOURCE[0]})"
echo -e "[INFO] ## ============================================== ##"
echo -e "[INFO]"

#
# globals
#

PROJECT_NAME="rklearn-lib"								# option = -p/--project
IMAGE_NAME="${PROJECT_NAME}"  								# option = -i/--image
HOST_PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."	# option = -d/--dir-host
IMAGE_VERSION="1.0.0"									# option = -v/--version-image

# the mounting point of the volume within the container
CONTAINER_PROJECT_DIR="/home/developer/workspace/${PROJECT_NAME}"    

############# 
## usage() ##
#############

function usage() {

cat << EOF
[INFO] Start a Docker container as a development env in a daemon mode. 
[INFO]
[INFO] Usage:
[INFO] <PROG> [-i <IMAGE_NAME> | --image=<IMAGE_NAME>] 
	[ -p <PROJECT_NAME> | --project=<PROJECT_NAME> ] 
	[ -d <HOST_PROJECT_DIR> | --dir-host=<HOST_PROJECT_DIR> ] 
	[ -v <IMAGE_VERSION> | --version-image=<IMAGE_VERSION> ] 
[INFO]
[INFO] Or: 
[INFO]
[INFO] <PROG> -h | --help
[INFO]
[INFO] Example:
[INFO]
[INFO] $ ./${0} -i IMAGE_NAME -p PROJECT_NAME -d PATH_TO_PROJECT_DIR -v IMAGE_VERSION
[INFO]
[INFO] Or with missing parameters to assign the default values:
[INFO]
[INFO] $ ./${0}
[INFO]
EOF

}

######################
## assign_default() ##
######################

function assign_default() {

	# hard coded values for this project

	[ -z "${PROJECT_NAME}" ] \
	    && PROJECT_NAME="rklearn-lib" \
	    && echo -e "[INFO] Automatically set the PROJECT_NAME = \e[32m${PROJECT_NAME}\e[39m" \
	|| echo -e "[INFO] PROJECT_NAME = \e[32m${PROJECT_NAME}\e[39m"

	[ -z "${IMAGE_NAME}" ] \
	    && IMAGE_NAME="${PROJECT_NAME}" \
	    && echo -e "[INFO] Automatically set the IMAGE_NAME = \e[32m${IMAGE_NAME}\e[39m" \
	|| echo -e "[INFO] IMAGE_NAME = \e[32m${IMAGE_NAME}\e[39m"

	[ -z "${IMAGE_VERSION}" ] \
	    && IMAGE_VERSION="1.0.0" \
	    && echo -e "[INFO] Automatically set the IMAGE_VERSION = \e[32m${IMAGE_VERSION}\e[39m" \
	|| echo -e "[INFO] IMAGE_VERSION = \e[32m${IMAGE_VERSION}\e[39m"

	[ ! -d "${HOST_PROJECT_DIR}" ] \
		&& echo -e "\e[31m[ERROR] Directory HOST_PROJECT_DIR = [${HOST_PROJECT_DIR}] does not exist on the host.\e[39m" \
		&& HOST_PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.." \
	   	&& echo -e "[INFO] Automatically set HOST_PROJECT_DIR = \e[32m${HOST_PROJECT_DIR}\e[39m" \
	|| echo -e "[INFO] HOST_PROJECT_DIR = \e[32m${HOST_PROJECT_DIR}\e[39m"

	CONTAINER_PROJECT_DIR="/home/developer/workspace/${PROJECT_NAME}"
	echo -e "[INFO] CONTAINER_PROJECT_DIR = \e[32m${CONTAINER_PROJECT_DIR}\e[39m"

}

##################
## parse_args() ##
##################

function parse_args() {

	# shorts = i,p,d,v,h 
	# long = --image, --project, --dir-host, --version-image, --help

	options=$(getopt -o hi:p:d:v: --long help --long image: --long project: --long dir-host: --long version-image: -- "$@")
	[ $? -eq 0 ] || { 
		usage 
		exit 1
	}
	eval set -- "$options"
	while true; do
		case "$1" in
		-d)
			shift;
			HOST_PROJECT_DIR=$1
			;;
		--dir-host)
			shift;
			HOST_PROJECT_DIR=$1
			;;
		-v)
			shift;
			IMAGE_VERSION=$1
			;;
		--version-image)
			shift;
			IMAGE_VERSION=$1
			;;
		-p)
			shift;
			PROJECT_NAME=$1
			;;
		--project)
			shift;
			PROJECT_NAME=$1
			;;
		-i)
			shift;
			IMAGE_NAME=$1
			;;
		--image)
			shift;
			IMAGE_NAME=$1
			;;
		-h)
			usage
			exit 1
			;;
		--help)
			usage
			exit 1
			;;
		--)
			shift
			break
			;;
		esac
		shift
	done # end while	

}

##########
## main ##
##########

# Parse the command line arguments
parse_args $0 "$@"

# Check assigned value
assign_default

echo -e ""
echo -e "[INPUT] Do you agree with these values (Y/n)?"
read response

[ -z "$response" ] && response="y" || true

[ "$response" == "y" ] || {

	echo -e "[WARN]"
	echo -e "[WARN] Aborting the process. Restart with appropriate options."
	echo -e "[WARN] Bye!"

	exit 1

}

echo -e "[INFO]"
echo -e "[INFO] Boostraping a container from the image ${IMAGE_NAME} using the command:"
echo -e "[INFO]"

set -x
docker run --runtime=nvidia --rm -d -it \
	--privileged \
	--name ${PROJECT_NAME} \
	-u developer \
	-v ${HOST_PROJECT_DIR}:${CONTAINER_PROJECT_DIR} \
	-v /dev/bus/usb:/dev/bus/usb \
	-e DISPLAY=$DISPLAY \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	${IMAGE_NAME}:${IMAGE_VERSION}
set +x

[ $? -eq 0 ] || { 
	echo -e "\e[31m[ERROR] Error during the docker run command execution! \e[39m"
	exit 1
}

cat << EOF
[INFO] 
[INFO] The container is correctly started in daemon mode:  
[INFO] - name = ${PROJECT_NAME}
[INFO] - network = host
[INFO] - user = developer
[INFO] - DISPLAY = $DISPLAY
[INFO]
[INFO] Access to the container through a terminal using the command:
[INFO] $ sudo docker exec -it ${PROJECT_NAME} /bin/bash
[INFO]
[INFO] At the end of the sseion, if necessary commit the container to a new image tag:
[INFO] $ sudo docker commit ${PROJECT_NAME} ${IMAGE_NAME}:<NEW VERSION> 
[INFO]
[INFO] Once commit, one can stop the container as follows:
[INFO] $ sudo docker stop ${PROJECT_NAME}
[INFO]
[INFO] Bye. 
EOF

exit 0


