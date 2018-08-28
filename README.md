# MAVI-mtp

## Runtime estimation and Preemption :  
### Face Detection:
- Dependencies: Opencv version > 2.4
- Build:``` g++ -o facevj My_face_detect_v1_demo.cpp loop_accl.cpp predictOrdered.cpp  `pkg-config opencv --cflags --libs` -std=c++11```
- Run:``` ./facevj input.jpg output.jpg timetointerruptinseconds ```
### Animal Detection:
-[ReadMe.txt]()

### Texture Detection:
- Build:``` g++ -std=c++11 glcm_test1.cpp -o new -I/usr/local/include -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs  -lopencv_videoio -lopencv_ml```
- Run:``` ./new inputimage outputimage```
### Pothole Detection:
- Build: ```make```
- Run: ```./a.out inputimage outputimage```


## For coexecution :
### Dependencies:
- openface 
- tensorflow and mobilenetssd model 

### Run:
- Edit input in coex.py
- ``` python coex.py switch ```
- switch : 
  - 1  -face 
  - 2 -animal
  - 3 -signboard
  - sa -sign+animal
  - sf - sign+face
  - af -animal+face
  - all -all together
