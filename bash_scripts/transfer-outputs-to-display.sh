#!/bin/bash

# Transfer output data
echo "Transfering output data!"
adb push /home/ff/Desktop/jetson-object-detection/zip_txt_final/* /storage/emulated/0/fireplay/camera-project/data
sudo rm -rf /home/ff/Desktop/jetson-object-detection/zip_txt_final/*

# Transfer logs
echo "Transfering logs!"
adb push /home/ff/Desktop/jetson-object-detection/zip_logs_final/* /storage/emulated/0/fireplay/camera-project/logs
sudo rm -rf /home/ff/Desktop/jetson-object-detection/zip_logs_final/*

# Transfer images data
echo "Transfering images!"
adb push /home/ff/Desktop/jetson-object-detection/zip_img_final/* /storage/emulated/0/fireplay/camera-project/image
sudo rm -rf /home/ff/Desktop/jetson-object-detection/zip_img_final/*

exit 0