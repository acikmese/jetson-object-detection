#!/bin/bash
echo "Updating Repo!"
adb pull /storage/emulated/0/fireplay/camera-project/repo/jetson-object-detection.zip /home/ff/Desktop/new_repo/
REPO=/home/ff/Desktop/new_repo/jetson-object-detection.zip
if [ -f "$REPO" ]; then
	# TODO: Add here stop the service of framework.
    unzip /home/ff/Desktop/new_repo/jetson-object-detection.zip -d /home/ff/Desktop/new_repo/
    rsync -az /home/ff/Desktop/new_repo/jetson-object-detection/ /home/ff/Desktop/jetson-object-detection/
    rm -rf /home/ff/Desktop/new_repo/jetson-object-detection
    rm -rf /home/ff/Desktop/new_repo/jetson-object-detection.zip
    # TODO: Add here to start the service again.
fi

exit 0