#!/bin/bash

# Found in following link.
# https://unix.stackexchange.com/a/205539

# Remove old (2 days) zipped data.
find /home/ff/jetson-object-detection/zipped_data/ -mindepth 1 -type f -mtime +2 -delete

# Remove old (12 hours) zipped images.
find /home/ff/jetson-object-detection/zipped_images/ -mindepth 1 -type f -mmin +720 -delete

# Remove old (2 days) zipped logs.
find /home/ff/jetson-object-detection/zipped_logs/ -mindepth 1 -type f -mtime +2 -delete
