#!/bin/bash

# Transfer output data
echo "Transferring output data!"
if adb push /home/ff/jetson-object-detection/zipped_data/*.zip /storage/emulated/0/fireplay/camera/data; then
  sudo rm -rf /home/ff/jetson-object-detection/zipped_data/*.zip
else
  echo "Could not transfer zipped data"
fi

# Transfer logs
echo "Transferring logs!"
if adb push /home/ff/jetson-object-detection/zipped_logs/*.zip /storage/emulated/0/fireplay/camera/log; then
  sudo rm -rf /home/ff/jetson-object-detection/zipped_logs/*.zip
else
  echo "Could not transfer zipped logs"
fi

# Transfer images data
echo "Transferring images!"
if adb push /home/ff/jetson-object-detection/zipped_images/*.zip /storage/emulated/0/fireplay/camera/picture; then
  sudo rm -rf /home/ff/jetson-object-detection/zipped_images/*.zip
else
  echo "Could not transfer zipped images"
fi
