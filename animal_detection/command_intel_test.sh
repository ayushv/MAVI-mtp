#!/bin/bash
g++ `pkg-config --cflags opencv` -c -g -MMD -MP -MF CPPdetection.o.d -o CPPdetection.o CPPdetection.cpp
g++ `pkg-config --cflags opencv` -o MAVI_AD CPPdetection.o svmlight/svm_learn.o svmlight/svm_hideo.o svmlight/svm_common.o `pkg-config --libs opencv`

