#include "opencv2/face.hpp" // Taken from Face recognition module
#include <fstream> // Taken from Face recognition module
#include <sstream> // Taken from Face recognition module
#include <chrono> // Taken from Face recognition module
#include <ctime> // Taken from Face recognition module
#include "loop.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui.hpp"
#include <cstdio>
#include <string>
#include "opencv2/core/core.hpp"
#include <map>
#include <deque>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::face;
template <typename T> string toString(T t)
{
    ostringstream out;
    out << t;
    return out.str();
}

void fr_recog(Mat test_face, int* label)
{
    /*Start*/
//	Size nsize(92,112); 										// the yml file is trained for image of size (92,112)
//	Size nsize(60,60); 										// the yml file is trained for image of size (92,112)
	Size nsize(70, 70);
//	Mat test_face; 											// Store the detected face pass it to recognition module

	double t3 = 0.0;
	t3 = (double)cvGetTickCount();
	Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer(); 					// Create a model for Fisher Face recognition
	//cout<<"Model Created"<<endl; 	
	//model -> load(argv[3]);										// Load the mode file (yml file) from the provided nput argument  
	model -> load("Trainning_set.yml");										// Load the mode file (yml file) from the provided nput argument  
//	model -> load("60x60.yml");										// Load the mode file (yml file) from the provided nput argument  
	//cout<<"Model Loaded"<<endl;
	double confidence = 0.0 ; 									// intial confidence is 0.0	
	int	predictedLabel = -1 ;									// Keep intially value of predicted lable as -1 (Labels are given -1 onwards)
//	test_face = image(r);										// Take Region of Interest in test_face
//	cout<<r<<endl;
	//cout<<"Size of the detected face"<<" "<<test_face.cols<<"x"<<test_face.rows<<endl;
	resize(test_face, test_face, nsize, 0, 0, INTER_CUBIC);						// Resize the detected face to 92x112
//	imshow("Resized Image", test_face);
//	waitKey(0);
	model -> predict(test_face, predictedLabel, confidence);					// actual prediction is done here	
	string outputStr;
	if(confidence > 1200){
	   outputStr = "\"Unknown Person\"";
	   *label = -1;
	}	 		
	else {
	   outputStr = toString(predictedLabel);
	   *label = predictedLabel;
	}	
	t3 = (double)cvGetTickCount() - t3;
//	printf("recognition time = %g (ms) \n", t3/((double) cvGetTickFrequency()*1000.));
	//cout <<"Predicted Index"<<" "<<outputStr<<" "<<"Prediction Confidence"<<" "<<confidence<<endl;
		
    //return 0;
}


