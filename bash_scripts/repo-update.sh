#!/bin/bash
echo "Updating Repo!"
# Pull new repo from display to jetson
adb pull /storage/emulated/0/fireplay/camera-project/repo/jetson-object-detection.zip /home/ff/Desktop/new_repo/
REPO=/home/ff/Desktop/new_repo/jetson-object-detection.zip
# Check if there is new repo file
if [ -f "$REPO" ]; then
    # Stop object detection service
	sudo systemctl stop firefly-object-detection.service
    unzip /home/ff/Desktop/new_repo/jetson-object-detection.zip -d /home/ff/Desktop/new_repo/
    # Copy new repo content to current object detection framework path
    rsync -az /home/ff/Desktop/new_repo/jetson-object-detection/ /home/ff/Desktop/jetson-object-detection/
    # Remove old files
    rm -rf /home/ff/Desktop/new_repo/jetson-object-detection
    rm -rf /home/ff/Desktop/new_repo/jetson-object-detection.zip
    # Restart camera service
    sudo systemctl restart nvargus-daemon
    # Start object detection service
    sudo systemctl start firefly-object-detection.service
fi

exit 0