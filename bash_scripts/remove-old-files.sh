#!/bin/bash

# Remove old zipped data.
# cd /home/ff/jetson-object-detection/zipped_data/
# ls *.zip -t | sed -e '1,500d' | xargs -d '\n' rm
find /home/ff/jetson-object-detection/zipped_data/ -mindepth 1 -type f -mmin +25 -delete

# Remove old zipped images.
# cd /home/ff/jetson-object-detection/zipped_images/
# ls *.zip -t | sed -e '1,100d' | xargs -d '\n' rm
find /home/ff/jetson-object-detection/zipped_images/ -mindepth 1 -type f -mmin +25 -delete

# Remove old zipped logs.
# cd /home/ff/jetson-object-detection/zipped_logs/
# ls *.zip -t | sed -e '1,500d' | xargs -d '\n' rm
find /home/ff/jetson-object-detection/zipped_logs/ -mindepth 1 -type f -mmin +25 -delete
