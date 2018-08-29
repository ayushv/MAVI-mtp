#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.h>
#include <opencv/ml.h>	
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
#include <iterator>
#include <cstdlib>

using namespace cv;
using namespace std;
int main(int argc, char const *argv[])
{
	/* code */
	Mat image;

	if (argc > 1) {
    image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  } else {
    cout << "Please input a image file\n";
    return 0;
  }


  Size size(image.cols/3,image.rows/3);
resize(image,image,size);

imwrite(argv[1],image);
  waitKey(0);

	return 0;
}