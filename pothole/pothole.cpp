#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencvblobslib/library/BlobResult.h"
#include <iostream>
#include <algorithm>
#include "sfta_new_80X80.hpp"
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
#include <clustering/SpectralClustering.h>
#include <cstdlib>
#include <csignal>
#include <ctime>
#include <sys/time.h>
#include <fstream>
#include "svm.h"
//#define INTERVAL  1000000
int INTERVAL;
int FEATURES = 24;
float mini[24], ranges[24];
struct svm_model *model;
svm_node *x_space;
Mat image,I;
vector<CBlob> boxs;
string output;
std::vector<double> timepts;
std::vector<double> estime;
double estimate;
double tickf=getTickFrequency();
int svm(Mat img) {
  double DT[FEATURES];
  sfta(img, DT);
  for (int i = 0 ; i < FEATURES ; i++) {
    DT[i] = (DT[i]-mini[i])/ranges[i];
  }
  for (int i = 0 ; i < FEATURES ; i++) {
    x_space[i].index = i;
    x_space[i].value = DT[i];
  }
  x_space[FEATURES].index = -1;
  return svm_predict(model, x_space);
}
double colsum(Mat mat, int col) {
  double res = 0;
  for (int i = 0 ; i < mat.rows ; i++) {
    res += mat.at<double>(i,col);
  }
  return res;
}
float icolsum(Mat mat, int col) {
    float res = 0;
    for (int i = 0 ; i < mat.rows ; i++) {
	res += mat.at<float>(i,col);
    }
    return res;
}
double colsumsq(Mat mat, int col) {
  double res = 0;
  for (int i = 0 ; i < mat.rows ; i++) {
    res += pow(mat.at<double>(i,col),2);
  }
  return res;
}
double rowsum(Mat mat, int row) {
  double res = 0;
  for (int i = 0 ; i < mat.cols ; i++) {
    res += mat.at<double>(row,i);
  }
  return res;
}
double rowsumsq(Mat mat, int s, int e, int row) {
  double res = 0;
  for (int i = s ; i < e ; i++) {
    res += pow(mat.at<double>(row,i), 2);
  }
  return res;
}
Eigen::MatrixXd calculateAffinity(Mat &image, Mat &hist) {
  float sigma = 0;
  for (int k = 0 ; k < 2 ; k++) {
    float up = 0, dn = 0;
    for (int i = 0 ; i < 256 ; i++) {
      float a = pow(hist.at<float>(i,k)-(1.0/256)*icolsum(hist,k),2);
      up = up+a;
      dn = dn+a;
    }
    up = up/255;
    dn = dn/255;
    dn = sqrt(dn);
    sigma = sigma+(up/dn);
  }
  sigma = sqrt(sigma);
  cout << "sigma" << sigma << endl;
  Eigen::MatrixXd affinity = Eigen::MatrixXd::Zero(256,256);
  for (unsigned int i = 0 ; i < 256 ; i++) {
    for (unsigned int j = 0 ; j < 256 ; j++) {
      double dist = sqrt(pow((double)(hist.at<float>(i,0)-hist.at<float>(j,0)),2)+pow((double)(hist.at<float>(i,1)-hist.at<float>(j,1)),2));
      affinity(i,j) = exp(-dist/(2*sigma*sigma));
    }
  }
  return affinity;
}
void bwareaopen(Mat img, double del) {
  CBlobResult blobs;
  blobs = CBlobResult(img,Mat(),4);
  blobs.Filter(blobs, B_INCLUDE, CBlobGetLength(), B_GREATER, del);
  img.setTo(0);
  for (int i = 0 ; i < blobs.GetNumBlobs(); i++) {
    blobs.GetBlob(i)->FillBlob(img, Scalar(255,255,255),0,0,true);
  }
}

void imcleanborder(Mat img,int radius) {
  for (int i = 0 ; i < img.cols ; i++) {

    floodFill(img, Point(i,0), Scalar(0),0,Scalar(), Scalar(),8);
    floodFill(img, Point(i,img.rows-1), Scalar(0),0,Scalar(), Scalar(),8);
  if(i==img.cols/3){

 	  timepts.push_back((double)getTickCount());
 	   estimate=100/10*(timepts.back()-timepts[0])/tickf; //avg 10%
 	  estime.push_back(estimate);
 	  cout<<"[10/100] Expected Runtime : "<<estimate<<endl;



  }  		
	if(i==img.cols*2/3){
 	  timepts.push_back((double)getTickCount());
 	   estimate=100/20*(timepts.back()-timepts[0])/tickf; //avg 20
 	  estime.push_back(estimate);
 	  cout<<"[20/100] Expected Runtime : "<<estimate<<endl;
  }    	

  }
  timepts.push_back((double)getTickCount());
   estimate=100/30*(timepts.back()-timepts[0])/tickf; //avg 30
 	estime.push_back(estimate);
 	  cout<<"[30/100] Expected Runtime : "<<estimate<<endl;
  for (int i = 0 ; i < img.rows ; i++) {
    floodFill(img, Point(0,i), Scalar(0),0,Scalar(),Scalar(),8);
    floodFill(img, Point(img.cols-1,i), Scalar(0),0,Scalar(),Scalar(),8);
  	if (i==img.rows/2)
  	{
  		  timepts.push_back((double)getTickCount());
  		   estimate=100/40*(timepts.back()-timepts[0])/tickf; //avg 40
 			estime.push_back(estimate);
 		  cout<<"[40/100] Expected Runtime : "<<estimate<<endl;
  	}
  }
    timepts.push_back((double)getTickCount());
     estimate=100/50*(timepts.back()-timepts[0])/tickf; //avg 50
 			estime.push_back(estimate);
 		  cout<<"[50/100] Expected Runtime : "<<estimate<<endl;

}
void removeLinearShapes(Mat img) {
  CBlobResult blobs;
  blobs = CBlobResult(img,Mat(),4);
  for (int i = 0 ; i < blobs.GetNumBlobs(); i++) {
    CBlob blob = blobs.GetBlob(i);
    double sq = sqrt((blob.Moment(2,0)-blob.Moment(0, 2))*(blob.Moment(2,0)-blob.Moment(0,2))+4*blob.Moment(1,1)*blob.Moment(1,1));
    double x = blob.Moment(2,0)+blob.Moment(0,2);
    double ecc = (x+sq)/(x-sq);
    cout << ecc << endl;
  }

}
double rowSum(double** mat, int size, int row) {
  double res = 0;
  for (int i = 0 ; i < size ; i++) {
    res += mat[row][i];
  }
  return res;
}










void signalHandler( int signum ) {
   cout << "Interrupt signal (" << signum << ") received.\n";
     cout << "size:" << boxs.size() << endl;

//   Mat I1(I.rows+2,I.cols+2,I.type()) ;

// for (int i = 0 ; i < boxs.size() ; i++) {
//     boxs.at(i).FillBlob(I1, Scalar(255,255,255),0,0,true);
//     Rect roi(boxs.at(i).GetBoundingBox().x, boxs.at(i).GetBoundingBox().y,boxs.at(i).GetBoundingBox().width, boxs.at(i).GetBoundingBox().height);
//     Mat Ss = image(roi);
//     Mat Sd;
//     resize(Ss, Sd, Size(80,80),0,0,INTER_CUBIC);
//     if (svm(Sd) > 0)
//       rectangle(image,Point(boxs.at(i).MinX(), boxs.at(i).MinY()), Point(boxs.at(i).MaxX(), boxs.at(i).MaxY()), Scalar(255,0,0));
//     else
//   rectangle(image,Point(boxs.at(i).MinX(), boxs.at(i).MinY()), Point(boxs.at(i).MaxX(), boxs.at(i).MaxY()), Scalar(0,0,0));
//   }



    imwrite(output,image);
    waitKey(0);

    ofstream myfile;
myfile.open ("outdata1.csv",ios::app);
myfile<<boxs.size()<<"\n";
myfile.close();

   // cleanup and close up stuff here  
   // terminate program  







   exit(signum);  
}







int main(int argc, char** argv) {

  timepts.push_back((double)getTickCount());
  output=argv[2];
  if (signal(SIGALRM, (void (*)(int)) signalHandler) == SIG_ERR) {
      perror("Unable to catch SIGALRM");
      exit(1);
    }
    signal(SIGINT, signalHandler); 
    signal(SIGTERM, signalHandler); 
    INTERVAL=(int)(atof(argv[3])*1000);
    struct itimerval it_val;
    it_val.it_value.tv_sec =     INTERVAL/1000;
    it_val.it_value.tv_usec =    (INTERVAL*1000) % 1000000;   
    it_val.it_interval = it_val.it_value;
    if (setitimer(ITIMER_REAL, &it_val, NULL) == -1) {
    perror("error calling setitimer()");
      exit(1);
    }
  

  //load svm 
  FILE *fp = fopen("pmini.csv", "r");
  for (int i = 0 ; i < FEATURES ; i++) 
    fscanf(fp, "%f", &mini[i]);
  fclose(fp);
  fp = fopen("pranges.csv", "r");
  for (int i = 0 ; i < FEATURES ; i++) 
    fscanf(fp, "%f", &ranges[i]);
  fclose(fp);
  model = svm_load_model("plibsvm");
  x_space = (svm_node*)malloc((FEATURES+1)*sizeof(svm_node));
  //
  /*VideoCapture capture("VID_0040.mp4");
  if (!capture.isOpened()) {
    throw "Error while opening video";
  }
  int imcount = 0;
  while(true) {*/
  Mat imgbw,img;
  if (argc > 1) {
    image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  } else {
    cout << "Please input a image file\n";
    return 0;
  }

//resize image

  //capture >> img;
  //if (img.empty())
    //break;
  //cvtColor(img,image,COLOR_BGR2GRAY);
  //imshow("image", image);
  //waitKey(0);



  double T = cv::threshold(image, imgbw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
  double t = T/255.0;
  cout << "Thr: " << t << endl;
  //imshow("binary", imgbw);
  //waitKey(0);
  cout << cv::sum(image) << endl;

  double del = abs(T-sum(image)[0]/(image.rows*image.cols))*2;
  if (del <= 10)
    del = 16;
  double T1 = (T+255)/2.0;
  cout << del << endl;
  Mat newimage = image.clone();
  cout << image.size() << endl;
  Mat c1 = newimage(cv::Rect(0,0,image.cols,image.rows-floor(del)));
  Mat c2 = newimage(cv::Rect(0,floor(del),image.cols,image.rows-floor(del)));
  cout << c1.size() << endl;
  cout << c2.size() << endl;
  /*imshow("c1", c1);
    waitKey(0);
    imshow("c2", c2);
    waitKey(0);*/
  Mat c = c1-c2;
  cv::threshold(c,c1,T1/4.0,255,CV_THRESH_BINARY);
  c = c1;
  imcleanborder(c,1);
  bwareaopen(c,del);
  //removeLinearShapes(c);
  Mat zrows(floor(del),c.cols,c.type());
  zrows.setTo(0);
  cv::vconcat(zrows,c,c);
  //imshow("c",c);
  //waitKey(0);

   

  int bins = 256;
  int histSize[] = {bins};
  float lranges[] = {0,256};
  const float* ranges[] = {lranges};
  cv::Mat hist;
  int channels[] = {0};
  cv::calcHist(&image,1,0,cv::Mat(),hist,1,histSize,ranges,true,false);
  Mat ohist = hist.clone();
  Mat tmp(hist.size(), hist.type());
  for (int i = 0 ; i < 256 ; i++) {
    tmp.at<float>(i,0) = i;
  }
  cv::hconcat(tmp,hist,hist);
  Eigen::MatrixXd affinity = calculateAffinity(image,hist);
  SpectralClustering* cl = new SpectralClustering(affinity, bins);
  Eigen::MatrixXd vals = cl->mEigenVal;
  cout << vals.size() << endl;
  double sumEigValues = 0;
  for (int i = 0 ; i < vals.size() ; i++) {
    sumEigValues += vals[i];
  }
  sumEigValues = sumEigValues/vals.size();
  double sumEigValues1 = 0;
  for (int i = 0 ; i < vals.size() ; i++) {
    sumEigValues1 += pow(vals[i]-sumEigValues,2);
  }
  sumEigValues1 = sumEigValues1/(vals.size()-1);
  sumEigValues = sqrt(sumEigValues1);
  int k = 0;
  for (int i = 0 ; i < vals.size() ; i++) {
    if (vals[i] > sumEigValues)
      k++;
  }
  std::vector<vector<int>> clusters = cl->clusterKmeans(k);
  I = image.clone();


  //imshow("I", I);
  //waitKey(0);
  //double pre1 = (double)getTickCount();


  for (int i = 0 ; i < k; i++) {
      cout << clusters[i].size() << endl;
    for (int j = 0 ; j < clusters[i].size() ; j++) {
      I.setTo(floor((i+1)*255.0/clusters.size()), image == clusters[i][j]);
    }
  }
  //imshow("I", I);
  //waitKey(0);


  Mat nzpoints;
  findNonZero(c, nzpoints);
  vector<Point> seedpts;
  vector<unsigned char> colors;
  for (int i = 0 ; i < nzpoints.total() ; i += 25) {
    seedpts.push_back(nzpoints.at<Point>(i));
    colors.push_back(I.at<uchar>(nzpoints.at<Point>(i)));
  }
  Mat I2(I.rows+2,I.cols+2,I.type()) ;
  I2.setTo(0);

  timepts.push_back((double)getTickCount());
estimate=100/60*(timepts.back()-timepts[0])/tickf; //avg 20
 			estime.push_back(estimate);
 		  cout<<"[60/100] Expected Runtime : "<<estimate<<endl;


  for(int i = 0 ; i < seedpts.size() ; i++) {
  	if(i==seedpts.size()/3){
  		timepts.push_back((double)getTickCount());
  		estimate=100/70*(timepts.back()-timepts[0])/tickf; //avg 20
 			estime.push_back(estimate);
 		  cout<<"[70/100] Expected Runtime : "<<estimate<<endl;

  	}

  	if(i==seedpts.size()*2/3){
  		  timepts.push_back((double)getTickCount());
  		estimate=100/80*(timepts.back()-timepts[0])/tickf; //avg 20
 			estime.push_back(estimate);
 		  cout<<"[80/100] Expected Runtime : "<<estimate<<endl;
  	}

    Point p = seedpts.at(i);
    if (p.x == -1)
      continue;
    uchar color = colors.at(i);
    seedpts.at(i) = Point(-1,-1);
    Mat I1 = I.clone();
    I1.setTo(0, I != color);
    Mat mask = Mat::zeros(I1.rows+2, I1.cols+2, CV_8U);
    floodFill(I1, mask, p, 255, 0, Scalar(), Scalar(), 4+(255<<8)+cv::FLOODFILL_MASK_ONLY);
    CBlobResult blobs;
    blobs = CBlobResult(mask,Mat(),4);
    if (blobs.GetNumBlobs() < 2)
      continue;
    CBlob blob = blobs.GetBlob(1);
    CvRect box = blob.GetBoundingBox();
    if (box.width <= del || box.height <= del || box.width >= image.cols*0.9 || box.height >= image.rows*0.9) 
      continue;
    for (int j = 0 ; j < seedpts.size() ; j++) {
      if (seedpts.at(j).x >= blob.MinX() && seedpts.at(j).x <= blob.MaxX() 
	  && seedpts.at(j).y >= blob.MinY() && seedpts.at(j).y <= blob.MaxY())
	seedpts.at(j) = Point(-1,-1);
    }
    boxs.push_back(blob);

    blob.FillBlob(I2, Scalar(255,255,255),0,0,true);
    Rect roi(blob.GetBoundingBox().x, blob.GetBoundingBox().y,blob.GetBoundingBox().width, blob.GetBoundingBox().height);
    Mat Ss = image(roi);
    Mat Sd;
    resize(Ss, Sd, Size(80,80),0,0,INTER_CUBIC);
    if (svm(Sd) > 0)
      rectangle(image,Point(blob.MinX(), blob.MinY()), Point(blob.MaxX(), blob.MaxY()), Scalar(255,0,0));
    else
  rectangle(image,Point(blob.MinX(), blob.MinY()), Point(blob.MaxX(), blob.MaxY()), Scalar(0,0,0));



  }

  timepts.push_back((double)getTickCount());
estimate=100/99*(timepts.back()-timepts[0])/tickf; //avg 20
 			estime.push_back(estimate);
 		  cout<<"[99/100] Expected Runtime : "<<estimate<<endl;

  cout << "size:" << boxs.size() << endl;
  
 

  /*while(!colors.empty()) {
    uchar color = colors.at(0);
    cout << color << endl;
    colors.erase(std::remove(colors.begin(), colors.end(), color), colors.end());
    //colors.remove(color);
    Mat I1 = I.clone();
    I1.setTo(0, I != color);

    CBlobResult blobs;
    cout << c.size() << endl;
    cout << I.size() << endl;
    blobs = CBlobResult(I1,c,4);
    cout << "NUM BLobs: " << blobs.GetNumBlobs() << endl;
    I1.setTo(0);
    for (int i = 0 ; i < blobs.GetNumBlobs(); i++) {
      if (blobs.GetBlob(i)->Area(PIXELWISE) <= del) {
	continue;
      }
      cout << "here it is" << endl;
      blobs.GetBlob(i)->FillBlob(I1, Scalar(255,255,255),0,0,true);
    }
  imshow("I",I1);
  waitKey(0);
  }*/
imwrite(output,image);
  waitKey(0);

  timepts.push_back((double)getTickCount());
double begin=timepts[0];
double total= timepts.back()-begin;
cout<<"total time : "<<total/tickf<<endl;
ofstream myfile;
myfile.open ("outdata.csv",ios::app);
myfile<<boxs.size()<<"\n";
myfile.close();




// ofstream myfile;
// myfile.open ("timedata3.csv",ios::app);
// myfile<<image.cols<<"x"<<image.rows<<",";
// myfile<<boxs.size() << ",";

// myfile<<seedpts.size() << ",";
// for (int i = 0; i < estime.size(); ++i)
// {	
// 	timepts[i]=(timepts[i]-begin)*100/total;
// 	cout<<timepts[i]<<endl;
// 	 myfile << timepts[i]<<",";

// }
// myfile<<"\n";
// myfile.close();

  /*stringstream ss;
  ss << imcount << ".png";
  cout << ss.str().c_str() << endl;
  imwrite(ss.str().c_str(), image);
  imcount++;
  }*/

  /*int const hist_height = 256;
  cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);
  double max_val = 0;
  minMaxLoc(ohist,0,&max_val);
  for (int b = 0 ; b < bins ; b++) {
    float const binVal = ohist.at<float>(b);
    int const height = cvRound(binVal*hist_height/max_val);
    cv::line(
	hist_image,
	cv::Point(b,hist_height-height), cv::Point(b,hist_height),
	find(clusters[0].begin(), clusters[0].end(), b) != clusters[0].end()?cv::Scalar::all(255):cv::Scalar::all(128)
	);
  }
  imshow("hist", hist_image);
  waitKey(0);*/
}
