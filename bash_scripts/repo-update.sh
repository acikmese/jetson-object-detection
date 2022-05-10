#!/bin/bash
echo "Updating Repo!"
adb pull /storage/emulated/0/fireplay/camera-project/repo/jetson-object-detection.zip /home/ff/Desktop/new_repo/
REPO=/home/ff/Desktop/new_repo/jetson-object-detection.zip
if [ -f "$REPO" ]; then
	sudo systemctl stop firefly-object-detection.service
    unzip /home/ff/Desktop/new_repo/jetson-object-detection.zip -d /home/ff/Desktop/new_repo/
    rsync -az /home/ff/Desktop/new_repo/jetson-object-detection/ /home/ff/Desktop/jetson-object-detection/
    rm -rf /home/ff/Desktop/new_repo/jetson-object-detection
    rm -rf /home/ff/Desktop/new_repo/jetson-object-detection.zip
    sudo systemctl start firefly-object-detection.service
fi

exit 0