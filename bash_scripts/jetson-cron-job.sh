#!/bin/bash

# 1. Transfer files to display path.
echo 1324 | sudo -S bash /home/ff/Desktop/jetson-object-detection/bash_scripts/transfer-outputs-to-display.sh
echo "Zipped files are transferred!"

# 2. Check repo update and do the update.
echo 1324 | sudo -S bash /home/ff/Desktop/jetson-object-detection/bash_scripts/repo_update.sh

# 3. Check if service is running, if not, restart.
echo 1324 | sudo -S bash /home/ff/Desktop/jetson-object-detection/bash_scripts/check-running-service.sh