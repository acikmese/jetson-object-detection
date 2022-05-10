#!/bin/bash

# Transfer output data
echo "Transferring output data!"
adb push /home/ff/Desktop/jetson-object-detection/zipped_data/* /storage/emulated/0/fireplay/camera-project/data
sudo rm -rf /home/ff/Desktop/jetson-object-detection/zipped_data/*

# Transfer logs
echo "Transferring logs!"
adb push /home/ff/Desktop/jetson-object-detection/zipped_logs/* /storage/emulated/0/fireplay/camera-project/logs
sudo rm -rf /home/ff/Desktop/jetson-object-detection/zipped_logs/*

# Transfer images data
echo "Transferring images!"
adb push /home/ff/Desktop/jetson-object-detection/zipped_images/* /storage/emulated/0/fireplay/camera-project/image
sudo rm -rf /home/ff/Desktop/jetson-object-detection/zipped_images/*

exit 0