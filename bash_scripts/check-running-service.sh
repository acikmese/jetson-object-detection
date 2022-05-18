#!/bin/bash

SERVICE="firefly-object-detection.service"
STATUS="$(systemctl is-active firefly-object-detection.service)"

# Check if service is active
if [ "${STATUS}" = "active" ]; then
    echo "$SERVICE is running!"
else
    echo " Service not running, restarting...! "
    # Restart camera daemon
    echo 1324 | sudo -S systemctl restart nvargus-daemon
    # Restart object detection service
    echo 1324 | sudo -S systemctl start firefly-object-detection.service
fi
