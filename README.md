# jetson-object-detection
Object detection framework for Nvidia Jetson Nano.  
Utilizes YoloV5 model.  
Works with 2 CSI cameras (also with USB and URL links) on Jetson Nano.  
Uses OpenCV for image processing.
Uses TensorRT optimized model for inference (done with export module from yolov5 framework).

## Installation

```bash
# Clone repo:  
git clone https://github.com/acikmese/jetson-object-detection.git
cd jetson-object-detection
# Initialize submodules:
git submodule init
git submodule update
# Setup pre-push hook:
# (Important for version update. It will warn you to push again.)
make pre-push-hook
