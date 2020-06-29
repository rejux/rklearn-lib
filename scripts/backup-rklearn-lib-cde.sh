#!/bin/bash


echo -e "[INFO]"
echo -e "[INFO] ## ============================================== ##"
echo -e "[INFO] ## $(basename ${BASH_SOURCE[0]})"
echo -e "[INFO] ## ============================================== ##"
echo -e "[INFO]"

#
# globals
#

# hard coded values for this project

IMAGE_VERSION="1.0.2"
PROJECT_NAME="rklearn-lib"
IMAGE_NAME="${PROJECT_NAME}"
PROJECT_DIR="/home/kla/workspace/data-science/ml-projects/${PROJECT_NAME}"

# The backup dir is generally put on a removable drive like ext usb disk
# BACKUP_DIR="/media/kla/multimedia/backups/docker-images/${PROJECT_NAME}"
BACKUP_DIR="/home/kla/docker-images/${PROJECT_NAME}"

#
# main
#

BACKUP_FILE=$(echo "${IMAGE_NAME}_${IMAGE_VERSION}.tar.gz" | tr "/" "_")

echo -e "[INFO]"
echo -e "[INFO] Automatically set PROJECT_NAME = \e[32m${PROJECT_NAME}\e[39m"
echo -e "[INFO] Automatically set PROJECT_DIR = \e[32m${PROJECT_DIR}\e[39m"
echo -e "[INFO] Automatically set IMAGE_NAME = \e[32m${IMAGE_NAME}\e[39m"
echo -e "[INFO] Automatically set IMAGE_VERSION = \e[32m${IMAGE_VERSION}\e[39m"
echo -e "[INFO] Automatically set BACKUP_DIR = \e[32m${BACKUP_DIR}\e[39m"
echo -e "[INFO] Automatically set BACKUP_FILE = \e[32m${BACKUP_FILE}\e[39m"
echo -e "[INFO]"

echo -e "[INPUT] Do you agree with these values (Y/n)?"
read response

[ -z "$response" ] && response="y" || true

[ "$response" == "y" ] || {

	echo -e "[WARN]"
	echo -e "[WARN] Aborting the process. Restart with appropriate options."
	echo -e "[WARN] Bye!"

	exit 1

}

# Create the backup infra

[ ! -d "${BACKUP_DIR}" ] || {

    # make a safe backup first
    rm -fr "${BACKUP_DIR}.old" || true
    mv "${BACKUP_DIR}" "${BACKUP_DIR}.old"

}

# collect info on the image:
packages_list=`docker run -it ${IMAGE_NAME}:${IMAGE_VERSION} dpkg -l`

mkdir -p ${BACKUP_DIR}
cat > "${BACKUP_DIR}/info.txt" << EOF

Meta data
---------

IMAGE_NAME = ${IMAGE_NAME}
IMAGE_VERSION = ${IMAGE_VERSION}
PROJECT_NAME = ${PROJECT_NAME}
PROJECT_DIR = ${PROJECT_DIR}

List of installed package
-------------------------

${packages_list}

EOF

# Save the image

echo -e "[INFO]"
echo -e "[INFO] Saving the image ${IMAGE_NAME}:${IMAGE_VERSION} to ${BACKUP_DIR}..."

# set -x
start_date="$(date)"
docker save "${IMAGE_NAME}:${IMAGE_VERSION}" | gzip > "${BACKUP_DIR}/${BACKUP_FILE}"
end_date="$(date)"
# set +x

# clean
[ -d "${BACKUP_DIR}.old" ] \
    && rm -fr "${BACKUP_DIR}.old" \
    || true


cat > "${BACKUP_DIR}/backup-report.txt" << EOF
start_date = ${start_date}
end_date = ${end_date}
backup file = ${BACKUP_DIR}/${BACKUP_FILE}
EOF

echo -e "[INFO] Done."
echo -e "[INFO]"

exit 0

