#!/bin/bash

SERVICE="firefly-object-detection.service"
STATUS="$(systemctl is-active firefly-object-detection.service)"

# Check if service is active
if [ "${STATUS}" = "active" ]; then
    echo "$SERVICE is running!"
else
    echo " Service not running, restarting...! "
    # Restart camera daemon
    sudo -S systemctl restart nvargus-daemon
    # Restart object detection service
    sudo -S systemctl restart firefly-object-detection.service
fi
