#!/bin/bash

# #umount /home/android
# sudo mkdir /home/ff/android_board
# sudo umount /home/ff/android_board
# sudo umount /run/user/1000/gvfs 
# sudo jmtpfs /home/ff/android_board/
# sleep 10

# sudo cp -r /home/ff/Desktop/jetson-object-detection/zip_txt_final /home/ff/android_board/fireplay/camera-project/data

# sudo ls -s 
# sudo cp -r "/home/ff/Desktop/jetson-object-detection/zip_txt_final" "/run/user/1000/gvfs/mtp:host=%5Busb%3A001%2C003%5D/fireplay/camera-project/data"

# sudo umount /home/ff/android_board
# sudo umount /run/user/1000/gvfs 
#sudo rm -rf /home/ff/android_board

adb push /home/ff/Desktop/jetson-object-detection/zip_txt_final/* /storage/emulated/0/fireplay/camera-project/data
sudo rm -rf /home/ff/Desktop/jetson-object-detection/zip_txt_final/*

exit 0

