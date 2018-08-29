#!/bin/bash
arm-linux-gnueabihf-g++-4.9 -std=c++0x CPPdetection.cpp -o arm_bin -I/home/baadalvm/OpenCV/build_arm_new/install/include/ -L/home/baadalvm/OpenCV/build_arm_new/all_libs/ -lopencv_objdetect -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -llibprotobuf -lIlmImf -llibjasper -llibjpeg -llibpng -llibwebp -lzlib -lpthread -llibtiff -ldl -lm -lrt
