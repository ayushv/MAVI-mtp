Training Hog+SVM:
-File: main.cpp
-Run ‘command_intel_train.sh’ or build by issuing the following commands:
  g++ `pkg-config --cflags opencv` -c -g -MMD -MP -MF main.o.d -o main.o main.cpp
  gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_learn.o.d -o svmlight/svm_learn.o svmlight/svm_learn.c
  gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_hideo.o.d -o svmlight/svm_hideo.o svmlight/svm_hideo.c
 gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_common.o.d -o svmlight/svm_common.o svmlight/svm_common.c
  g++ `pkg-config --cflags opencv` -o trainhog main.o svmlight/svm_learn.o svmlight/svm_hideo.o svmlight/svm_common.o `pkg-config --libs opencv`


-Run by issuing: 
./trainhog


Testing Hog+SVM:
-File: CPPdetection.cpp
-Run ‘command_intel_train.sh’ or build by issuing the following commands to compile the code:
g++ `pkg-config --cflags opencv` -c -g -MMD -MP -MF CPPdetection.o.d -o CPPdetection.o CPPdetection.cpp
g++ `pkg-config --cflags opencv` -o MAVI_AD CPPdetection.o svmlight/svm_learn.o svmlight/svm_hideo.o svmlight/svm_common.o `pkg-config --libs opencv`
-Run by issuing:
 ./MAVI_AD <image name>


Cross-compiling for Zedboard:
-Run the script command_arm.sh