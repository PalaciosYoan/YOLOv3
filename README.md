# YOLOv3 using Python (3.9) opencv-python

**Overview: ** This was a fun little project to implement YOLOv3 using Python cv.

**You need the following:**
- Camera connected to your PC, such as a webcam.
- install numpy and opencv-python

**If you would like to use this you will need to download the following:**
- You need to download the weights for YOLOc3-320 at https://pjreddie.com/darknet/yolo/
  - Scroll down to Performance on the COCO Dataset and look for YOLOv3-320 and click on weights to download them.
- The config file for YOLOv3-320 is already provided in the zip file

**After cloning the repo:**
- change file path location in line 9,16,17 to your corresponding file path location for the coco.names, config file, and weights file

After completing the steps above you should be ready to run YOLOv3! 

To terminate the program just click ctrl+c, forgot to add a quit feature for it.

**You can improve the speed by...**
- Download the config and weights file for YOLOv3-tiny (https://pjreddie.com/darknet/yolo/)
- Replace the 320 config and weight files with tiny's config and weights files.
