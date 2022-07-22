#!/bin/bash

# Give permissions
sudo chmod 777 -R /home/ff/jetson-object-detection
sudo chmod 777 -R /home/ff/tmp_repo_dir

# 1. Zip cron log and transfer to zipped logs.
zip /home/ff/jetson-object-detection/zipped_logs/cron_log_"`date +"%s"`".zip . -i /home/ff/jetson-object-detection$
truncate -s 0 /home/ff/jetson-object-detection/cron_log/cron_log.log  # empty file

# 2. Transfer files to display path.
sudo -S bash /home/ff/jetson-object-detection/bash_scripts/transfer-outputs-to-display.sh
echo "Transferring files are done!"

# 3. Check repo update and do the update.
sudo -S bash /home/ff/jetson-object-detection/bash_scripts/repo-update.sh
echo "Repo update is done!"

# 4. Check if service is running, if not, restart.
sudo -S bash /home/ff/jetson-object-detection/bash_scripts/check-running-service.sh
echo "Service check is done!"

# 5. Remove old files if necessary.
sudo -S bash /home/ff/jetson-object-detection/bash_scripts/remove-old-files.sh
echo "Old files are removed!"

exit 0