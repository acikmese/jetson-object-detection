#!/bin/bash

echo "Updating Repo!"

# Pull new repo from display to jetson
adb pull /storage/emulated/0/fireplay/camera/source/jetson-object-detection.zip /home/ff/tmp_repo_dir/
REPO=/home/ff/tmp_repo_dir/jetson-object-detection.zip

# Check if there is new repo file
if [ -f "$REPO" ]; then
    echo "New repo pulled, changing files!"

    # Stop object detection service
    echo 1324 | sudo -S systemctl stop firefly-object-detection.service
    echo "Object detection service is stopped!"

    # Unzip to directory
    unzip /home/ff/tmp_repo_dir/jetson-object-detection.zip -d /home/ff/tmp_repo_dir/
    echo "Files are unzipped!"

    # Copy new repo content to current object detection framework path
    rsync -az /home/ff/tmp_repo_dir/jetson-object-detection/ /home/ff/jetson-object-detection/
    echo "Repo files are changed!"

    # Remove old files
    rm -rf /home/ff/tmp_repo_dir/jetson-object-detection
    rm -rf /home/ff/tmp_repo_dir/jetson-object-detection.zip
    echo "Old files are removed!"

    # Restart camera service
    echo 1324 | sudo -S systemctl restart nvargus-daemon
    echo "nvargus-daemon service is restarted!"

    # Start object detection service
    echo 1324 | sudo -S systemctl start firefly-object-detection.service
    echo "Object detection service is started!"

    # Remove repo file from display.
    adb shell rm /storage/emulated/0/fireplay/camera/source/jetson-object-detection.zip
    echo "Old repo file is deleted from display!"
fi

exit 0
