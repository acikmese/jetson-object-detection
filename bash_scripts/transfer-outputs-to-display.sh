#!/bin/bash

# Transfer output data
echo "Transferring output data!"
adb push /home/ff/jetson-object-detection/zipped_data/*.zip /storage/emulated/0/fireplay/camera/data
sudo rm -rf /home/ff/jetson-object-detection/zipped_data/*.zip

# Transfer logs
echo "Transferring logs!"
adb push /home/ff/jetson-object-detection/zipped_logs/*.zip /storage/emulated/0/fireplay/camera/log
sudo rm -rf /home/ff/jetson-object-detection/zipped_logs/*.zip

# Transfer images data
echo "Transferring images!"
adb push /home/ff/jetson-object-detection/zipped_images/*.zip /storage/emulated/0/fireplay/camera/picture
sudo rm -rf /home/ff/jetson-object-detection/zipped_images/*.zip

exit 0
