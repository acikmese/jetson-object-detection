#!/bin/bash

echo "Updating Repo!"

DISPLAY_FILE_PATH=$(adb shell ls /storage/emulated/0/fireplay/camera/source/source*.zip | head -1)
DISPLAY_FILE="$(basename -- $DISPLAY_FILE_PATH)"
LOCAL_FILE_PATH=$(ls /home/ff/tmp_repo_dir/source*.zip | head -1)
LOCAL_FILE="$(basename -- $LOCAL_FILE_PATH)"

if [ -z "$DISPLAY_FILE_PATH" ] || [ "$LOCAL_FILE" = "$DISPLAY_FILE" ]; then
    echo "No need to update source repo!"
else
    echo "New source file found, updating source repo!"
    NEW_LOCAL_PATH="/home/ff/tmp_repo_dir/"$DISPLAY_FILE""

    echo "Removing old local file."
    rm -rf $LOCAL_FILE_PATH

    echo "Pull new repo!"
    adb pull $DISPLAY_FILE_PATH /home/ff/tmp_repo_dir/
    echo "New repo pulled, changing files!"

    # Stop object detection service
    echo 1324 | sudo -S systemctl stop firefly-object-detection.service
    echo "Object detection service is stopped!"

    # Unzip to directory
    unzip $NEW_LOCAL_PATH -d /home/ff/tmp_repo_dir/
    echo "Files are unzipped!"

    # Copy new repo content to current object detection framework path
    rsync -az /home/ff/tmp_repo_dir/jetson-object-detection/ /home/ff/jetson-object-detection/
    echo "Repo files are changed!"

    # Remove old directory
    rm -rf /home/ff/tmp_repo_dir/jetson-object-detection
    echo "Old directory is removed!"

    # Restart camera service
    echo 1324 | sudo -S systemctl restart nvargus-daemon
    echo "nvargus-daemon service is restarted!"

    # Start object detection service
    echo 1324 | sudo -S systemctl start firefly-object-detection.service
    echo "Object detection service is started!"
fi
