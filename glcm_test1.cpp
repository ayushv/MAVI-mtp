#include <iostream>
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdlib>
#include <cmath>
#include <string>
#include <csignal>
#include <ctime>
#include <sys/time.h>
#include <fstream>
//#define INTERVAL 200	
using namespace cv; using namespace std; using namespace cv::ml;	
static double ASM,contrast, correlation, IDM, entropy, dissimilarity,px,py;
Mat img,img1;
int pred[84];
Mat overlay;
string output;
int k;
int INTERVAL;
std::vector<double> times;
void greycomatrix(Mat img)
{
	int sum=0, i,a,b;
	double glcm[257][257]={0.0},stdevx=0.0,stdevy=0.0, sx=0.0,sy=0.0,cov=0.0;
	ASM=0.0,contrast=0.0, px=0.0,py=0.0, correlation=0.0, IDM=0.0, entropy=0.0, dissimilarity=0.0;
	for (int x=0; x<img.rows; x++)
	{
		for (int y=0; y<img.cols-1; y++)
		{
			a=0xff & img.at<uchar>(x,y);
			b=0xff & img.at<uchar>(x,y+1);
			glcm [a][b] +=1;
			glcm [b][a] +=1;
			sum+=2;
		}
	}
	for (a=0;  a<257; a++)  {
		for (b=0; b<257;b++) {
			glcm[a][b]/=sum;
			ASM+= (glcm[a][b]*glcm[a][b]);
			contrast+= ((a-b)*(a-b)*glcm[a][b]);
			dissimilarity+= abs(a-b)*glcm[a][b];
			px+=a*glcm [a][b];
		        py+=b*glcm [a][b];
            		IDM=IDM+ (glcm[a][b]/(1+(a-b)*(a-b)))  ;
           		 if (glcm[a][b]!=0)
            			entropy-=(glcm[a][b]*(log10(glcm[a][b])));
		}
	}
	for (a=0; a<257; a++){
		for (b=0; b <257; b++){
			stdevx+=(a-px)*(a-px)*glcm [a][b];
			stdevy+=(b-py)*(b-py)*glcm [a][b];
			cov+=(a-px)*(b-py)*glcm[a][b];
		}
	}
	sx=sqrt(stdevx);
	sy=sqrt(stdevy);
	correlation=cov/(sx*sy);
}


void signalHandler( int signum ) {
   cout << "Interrupt signal (" << signum << ") received.\n";

	imwrite(output,img1);
	waitKey(0);
	 ofstream myfile;
myfile.open ("preout.csv",ios::app);
myfile<<k<<"\n";
myfile.close();
   // cleanup and close up stuff here  
   // terminate program  

   exit(signum);  
}
void termtimer(){

	struct itimerval it_val;  /* for setting itimer */

  /* Upon SIGALRM, call DoStuff().
   * Set interval timer.  We want frequency in ms, 
   * but the setitimer call needs seconds and useconds. */
  if (signal(SIGALRM, (void (*)(int)) signalHandler) == SIG_ERR) {
    perror("Unable to catch SIGALRM");
    exit(1);
  }
  it_val.it_value.tv_sec =     INTERVAL/1000;
  it_val.it_value.tv_usec =    (INTERVAL*1000) % 1000000;   
  it_val.it_interval = it_val.it_value;
  if (setitimer(ITIMER_REAL, &it_val, NULL) == -1) {
 	perror("error calling setitimer()");
    exit(1);
  }}

int main ( int argc, char *argv[] ) {
//Arguments
	//1.Image
double begin = (double)getTickCount();
double tickf=getTickFrequency(); 		

//clock_t begin = clock();
    INTERVAL=(int)(atof(argv[3])*1000);

termtimer();
if (argc == 1)
{
	return 0;
}
output=argv[2];
Ptr<SVM> svm = StatModel::load<SVM>("glcm_model.xml");
Mat test(1,9,CV_32F);

img = imread(argv[1], 0);
img1=imread(argv[1]);
Size s(640,480);
resize(img,img,s);
k=0;
overlay=img1.clone();
signal(SIGINT, signalHandler); 
signal(SIGTERM, signalHandler); 

double loops = (double)getTickCount();
double runtime=100/2.8*(loops-begin)/tickf;
cout<<"[3/100] Expected Runtime:"<<runtime<<endl;
times.push_back(runtime);
for(int i=5;i<12;i++)
{
	double in=(double)getTickCount();
	for(int j=2;j<14;j++)
	{
		
		Mat block=img(Rect(Point(j*40,i*40),Point((j+1)*40,(i+1)*40)));
		greycomatrix(block);
		test.at<float>(0,0)=ASM;
		test.at<float>(0,1)=contrast;
		test.at<float>(0,2)=correlation;
		test.at<float>(0,3)=IDM;
		test.at<float>(0,4)=entropy;
		test.at<float>(0,5)=dissimilarity;
		test.at<float>(0,6)=sqrt(ASM);
		test.at<float>(0,7)=px;
		test.at<float>(0,8)=py;
		pred[k]= svm->predict(test);
		Mat roi = img1(Rect(j*40, i*40, 40,40));
		Mat color(roi.size(), CV_8UC3, Scalar(255, 255, 0));
		if(pred[k]==1.0)
				color=Scalar(0,0,255);	//Pavement
		else if(pred[k]==2.0)
			color=Scalar(255, 0, 0); 		//Road
    	
    	double alpha = 0.3;
    	addWeighted(color, alpha, roi, 1.0 - alpha , 0.0, roi); 



		k++;
		
		imwrite(argv[2],img1);	
	}
	runtime=100*(((double)getTickCount()-begin))/((13.57*(i-4)+3)*tickf);
	cout<<"["<<3+(i-4)*13<<"/100] Expected Runtime:"<<runtime<<endl;
	times.push_back(runtime);
}
double loope=(double)getTickCount();

// 
// cout<<t<<endl;
// for(int i=5;i<12;i++)
// {
//    for(int j=2;j<14;j++)
//    { 
//    		if(pred[(i-5)*12+(j-2)]==1.0)
//    			cout<<"bottom-left-"<<j*40<<","<<i*40<<"\ttop-right-"<<(j+1)*40<<","<<(i+1)*40<<"\tLabel-Pavement"<<endl;
//    		else if(pred[(i-5)*12+(j-2)]==2.0)
//    			cout<<"bottom-left-"<<j*40<<","<<i*40<<"\ttop-right-"<<(j+1)*40<<","<<(i+1)*40<<"\tLabel-Road"<<endl;
//    		else
//    			cout<<"bottom-left-"<<j*40<<","<<i*40<<"\ttop-right-"<<(j+1)*40<<","<<(i+1)*40<<"\tLabel-Others"<<endl;
//    	}
// }
// for(int i=5;i<12;i++)
// {
//    for(int j=2;j<14;j++)
//    { 
//     	if(pred[(i-5)*12+(j-2)]==1.0)
// 		rectangle(overlay, Point(j*40, i*40), Point((j+1)*40,(i+1)*40),Scalar(0, 0, 255),-1);		//Pavement
// 	else if(pred[(i-5)*12+(j-2)]==2.0)
// 		rectangle(overlay, Point(j*40, i*40), Point((j+1)*40,(i+1)*40),Scalar(255, 0, 0),-1);		//Road
// 	else
// 		rectangle(overlay, Point(j*40, i*40), Point((j+1)*40,(i+1)*40),Scalar(255, 255,0),-1);		//Others
//     }
// }
// addWeighted(overlay, 0.4, img1, 0.6,0, img1);

imwrite(argv[2],img1);
cout<<"Writing image to "<<argv[2]<<endl;
double end =(double)getTickCount();
 double total = (end - begin)/tickf; 
cout<<"total time : "<<total<<endl;
// cout<<"start to loop start :"<<(loops-begin)/tickf*280<<" %:" << ((loops-begin)/tickf)*100/total<<endl;
 cout<<"loop time : "<<(loope-loops)/tickf*100/total<<endl;
// cout<<"end time : "<<(end-loope)/tickf*100/total<<endl;
//  ofstream myfile;
// myfile.open ("estimates.csv",ios::app);

// for (int i = 0; i < times.size(); ++i)
// {
// 	myfile<<times[i]<<",";
// }
// myfile<<total<<"\n";
// myfile.close();

return 0;
}
