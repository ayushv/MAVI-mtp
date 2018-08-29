#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <algorithm>
#include "sfta_new_80X80.hpp"
#include "svm.h"
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

int C =3100;
int FEATURES = 24;
#define WORKING_DIR "/mnt/disk1/durgesh_home/Pothole/"
#define Malloc(type, n) = (type *)malloc((n)*sizeof(type))
using namespace cv::ml;
using namespace std;
using namespace cv;
struct svm_parameter param;
struct svm_problem prob;
struct svm_model *model;
string exec(const char* cmd) {
  char buffer[128];
  string result = "";
  shared_ptr<FILE> pipe(popen(cmd,"r"),pclose);
  if (!pipe) throw runtime_error("popen() failed");
  while(!feof(pipe.get())) {
    if (fgets(buffer, 128, pipe.get()) != NULL)
      result += buffer;
  }
  return result;
}

// accuracy
float evaluate(cv::Mat& predicted, cv::Mat& actual) {
  assert(predicted.rows == actual.rows);
  int t = 0;
  int f = 0;
  for(int i = 0; i < actual.rows; i++) {
    int p = predicted.at<int>(i,0);
    int a = actual.at<int>(i,0);
    //cout<<"p"<<p<<"  ";
    //cout<<"a"<<a<<endl;

    //        if((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) 
    if((p==a)) {

      t++;
    } else {
      f++;
    }
  }
  return (t*1.0)/(t+f);
}


int main( int argc, char** argv )
{
  Mat image, Data,AT, Crop, labels_pave, Actual_pave;
  string line;
  ifstream imagefile;
  ifstream labelfile;
  string dbRoot;
  dbRoot.append(WORKING_DIR);
  dbRoot.append("DATA_BASE/pothole_data/");
  string indexFile;
  indexFile.append(dbRoot);
  indexFile.append("index.txt");
  string xmlRoot;
  xmlRoot.append(WORKING_DIR);
  xmlRoot.append("Results/XML/80X80/");
  string resultXML;
  resultXML.append(xmlRoot);
  resultXML.append("pave_80X80.xml");
  string labelFile;
  labelFile.append(dbRoot);
  labelFile.append("lables.txt");
  
  double *d;
  double DT[FEATURES];
  float DDT[C][FEATURES];
  float mini[FEATURES];
  float ranges[FEATURES];
  int W1=80;
  int W2=80;
  int l,k,m;
  int P[C], Q[C];
  /*----Setting training data----*/
  imagefile.open (indexFile.c_str());
  int c=0;
  int line_count = 0;
  labelfile.open(labelFile.c_str());
  string line1;

  while (getline(imagefile,line,'\n'))
  {
    line.insert(0,dbRoot);
    image = imread(line , 0);
    /*String cmd = "~/mavi/caffe/cmake_build/examples/cpp_classification/classification testval.prototxt caffenet_train_iter_700.caffemodel a b ";
    cmd = cmd + line;
    stringstream ss(exec(cmd.c_str()));
    for (int i = 0 ; i < FEATURES ; i++) {
      ss >> DT[i];
    }*/
    getline(labelfile, line1, '\n'); 
    cout << line << " " << line1 << endl;
    //Size size(80,80);
    //resize(image,image,size);
    cout << c << "<->" << line << endl;
    //for (int x = 0 ; x < 480 ; x += W2) {
      //for (int y = 0 ; y < 640 ; y+= W1) {
	//Rect myROIT1(y, x, W1, W2);
	Crop = image;//(myROIT1);
	sfta(Crop,DT);
	//int l = floor(x/W2);
	//int k = ceil(y/W1);
	//c = (line_count)*((640*480)/(W1*W2))+l*(640/W1)+k;

	//cout<<c<<endl;
	//getchar();
	for(int i=0;i<FEATURES;i++)
	{	
	  DDT[c][i]=(float) DT[i];
	  if (c == 0){
	    mini[i] = DT[i];
	    ranges[i] = DT[i];
	  } else {
	    if (mini[i] > DT[i]) 
	      mini[i] = DT[i];
	    if (ranges[i] < DT[i]) 
	      ranges[i] = DT[i];
	  }
	  cout<<DT[i]<<" ";			
	}
	cout << endl << endl << endl;
	if (line1.compare("p") == 0) {
	  P[c] = 1;
	} else if(line1.compare("r") == 0){
	  P[c] = 2;
	} else if(line1.compare("g") == 0) {
	  P[c] = 3;
	} else {
	  P[c] = -1;
	}
      //}
    //}
    c++;
    line_count += 1;
  }
  imagefile.close();
  for (int i = 0 ; i < FEATURES ; i++) {
    ranges[i] = ranges[i]-mini[i];
  }

  for (int i = 0 ; i < c ; i++) {
    for (int j = 0 ; j < FEATURES ; j++) {
      if (ranges[j] != 0)
      	DDT[i][j] = (DDT[i][j]-mini[j])/ranges[j];
      else
	cout << "Range is zero" << endl;
    }
  }

  param.degree = 3;
  param.coef0 = 0;
  param.nu = 0.5;
  param.cache_size = 100;
  param.eps = 1e-8;
  param.p = 0.1;
  param.shrinking = 1;
  param.probability = 1;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
  prob.l = c;
  svm_node** x = (svm_node**)malloc((prob.l)*sizeof(svm_node*));
  for (int i = 0 ; i < c ; i++) {
    svm_node *x_space = (svm_node*)malloc((FEATURES+1)*sizeof(svm_node));
    for (int j = 0 ; j < FEATURES ; j++) {
      x_space[j].index = j;
      x_space[j].value = DDT[i][j];
    }
    x_space[FEATURES].index = -1;
    x[i] = x_space;
  }
  prob.x = x;
  prob.y = (double*)malloc(prob.l*sizeof(double));
  for (int i = 0 ; i < c ; i++) {
    prob.y[i] = P[i];
  }
  FILE *pfp = fopen("psvm_prob_all_test", "w");
  FILE *lfp = fopen("plabes_all_test.csv", "w");
  FILE *dfp = fopen("pdata_all_test.csv", "w");
  FILE *rfp = fopen("pranges_all_test.csv", "w");
  FILE *mfp = fopen("pmini_all_test.csv", "w");
  for (int i = 0 ; i < c ; i++) {
    fprintf(pfp,"%d ", P[i]);
    for (int j = 0 ; j < FEATURES ; j++) {
      fprintf(pfp, "%d:%f", j+1,DDT[i][j]);
      if (j < FEATURES-1)
	fprintf(pfp, " ");
      else
	fprintf(pfp, "\n");
    }
  }
  for (int i = 0 ; i < FEATURES ; i++) {
    fprintf(rfp, "%f\n", ranges[i]);
    fprintf(mfp, "%f\n", mini[i]);
  }
  fclose(pfp);
  fclose(lfp);
  fclose(dfp);
  fclose(rfp);
  fclose(mfp);
  //Data = Mat(c, 24, CV_32FC1, DDT );
  //cout<<Data<<endl;

  /*-----------------------------------------------------------Labels-----------------------------------------------------------------*/
  //labels_pave = Mat(c, 1, CV_32SC1, P ); //for 1D array
  //cout << c << " " << Data.rows << "-" << Data.cols << " " << labels_pave.rows << "-" << labels_pave.cols << endl;
  /*------------------------------------------------------------Training---------------------------------------------------------------*/

  cout << "Starting Pave training process" << endl;

  param.svm_type = C_SVC;
  param.kernel_type = LINEAR;
  param.gamma = 1;
  param.C = 1;
  model = svm_train(&prob, &param);
  svm_save_model("plibsvm_all_test", model);
  //! [init]
  /*  Ptr<SVM> svm_pave = SVM::create();
      svm_pave->setType(SVM::C_SVC);
      svm_pave->setC(1);
      svm_pave->setGamma(1);
  //svm_pave->setNu(0.8);
  //svm_pave->setDegree(2);
  svm_pave->setKernel(SVM::LINEAR);
  svm_pave->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));
  //! [init]
  //! [train]
  svm_pave->train(Data, cv::ml::ROW_SAMPLE, labels_pave);
  //! [train] */
  cout << "Finished Pave training process" << endl;

  //svm_pave->save("pave_80X80.xml");

  /*--------------------------------------------------------------Predicting--*-------------------------------------------------*
    cout << "starting testing..." << endl;
  //cout << resultXML.c_str() << endl;
  svm_pave  = Algorithm::load<SVM>("pave_80X80.xml");

  imagefile.open (indexFile.c_str());
  c=0;
  int inc=0;
  while ( getline (imagefile,line,'\n'))
  {
  line.insert(0,dbRoot);
  //    cout << line << endl;
  image = imread(line , 0);
  Size size(200,200);
  resize(image,image,size);
  sfta(image,DP);
  //cout<<m<<endl;
  //getchar();
  for(int i=0;i<24;i++)
  {	
  DDP[c][i]=(float) DP[i];
  DDP[c][i] = (DDP[c][i]-mini[i])/ranges[i];
  //cout<<D[i]<<" ";			
  }
  AT = Mat(1, 24, CV_32FC1, DDP[c] );
  int res_pave = svm_pave->predict(AT);
  //    cout<<res_pave<<endl;
  Q[c]=res_pave;

  c++;
  }

  imagefile.close();

  Actual_pave = Mat(c, 1, CV_32SC1, Q ); //for 1D array
  //cout<<"true"<<labels_pave<<endl;
  //cout<<"true"<<Actual_pave<<endl;
  cout << "Accuracy_{SVM}_80X80 = " << evaluate(Actual_pave, labels_pave) << endl;*/

  return 0;   
}

