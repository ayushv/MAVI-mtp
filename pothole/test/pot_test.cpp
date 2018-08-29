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

int C=1500;
int FEATURES=24;
#define WORKING_DIR "/home/baadalvm/Desktop/mavi/Pothole/"
#define Malloc(type, n) = (type *)malloc((n)*sizeof(type))
using namespace cv::ml;
using namespace std;
using namespace cv;
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

int main( int argc, char** argv )
{
  string line;
  Mat image, Crop;
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

  double DT[FEATURES];
  float DDT[C][FEATURES];
  int P[C];
  float mini[FEATURES], ranges[FEATURES];
  int l,m,k;
  imagefile.open (indexFile.c_str());
  int c=0;
  int line_count = 0;
  labelfile.open(labelFile.c_str());
  string line1;

  while (getline(imagefile,line,'\n'))
  {
    line.insert(0,dbRoot);
    image = imread(line , 0);
    getline(labelfile, line1, '\n'); 
    cout << c << "<->" << line << endl;
    /*String cmd = "~/mavi/caffe/cmake_build/examples/cpp_classification/classification testval.prototxt caffenet_train_iter_700.caffemodel a b ";
    cmd = cmd + line;
    stringstream ss(exec(cmd.c_str()));
    for (int i = 0 ; i < FEATURES ; i++) {
      ss >> DT[i];
    }
    cout << cmd << " " << line1 << endl;*/
    Crop = image;//(myROIT1);
    sfta(Crop,DT);
    for(int i=0;i<FEATURES;i++)
    {	
      DDT[c][i]=(float) DT[i];
      //cout<<DT[i]<<" ";			
    }
    if (line1.compare("p") == 0) {
      P[c] = 1;
    } else if(line1.compare("r") == 0){
      P[c] = 2;
    } else if(line1.compare("g") == 0) {
      P[c] = 3;
    } else {
      P[c] = -1;
    }
    c++;
    line_count += 1;
  }
  imagefile.close();


  FILE *fp = fopen("pmini_all_test.csv", "r");
  for (int i = 0 ; i < FEATURES ; i++)
    fscanf(fp,"%f",&mini[i]);
  fclose(fp);
  fp = fopen("pranges_all_test.csv", "r");
  for (int i = 0 ; i < FEATURES ; i++) {
    fscanf(fp, "%f", &ranges[i]);
  }
  fclose(fp);

  model = svm_load_model("plibsvm_all_test");
  svm_node *x_space = (svm_node*)malloc((FEATURES+1)*sizeof(svm_node));
  int acq = 0, total = 0;
  /*int cmatrix[4][4];
  for (int i = 0 ; i < 4 ; i++) {
      for (int j = 0 ; j < 4 ; j++) {
	  cmatrix[i][j] = 0;
      }
  }*/
  for (int i=0;i<c;i++)
  {
    for (int x = 0 ; x < FEATURES ; x++) {
      DDT[i][x] = (DDT[i][x]-mini[x])/ranges[x];
    }
    for (int j = 0 ; j < FEATURES ; j++) {
      x_space[j].index = j;
      x_space[j].value = DDT[i][j];
    }
    x_space[FEATURES].index = -1;
    int res_pave = svm_predict(model,x_space);
    cout << res_pave << "<-->" << P[i] << endl;
    if (res_pave == P[i]) 
	acq++;
    //cmatrix[P[i]>0?P[i]:0][res_pave>0?res_pave:0]++;
    total++;
  }
  cout << acq*100/total << endl;
  /*for (int i = 0 ; i < 4 ; i++) {
      for (int j = 0 ; j < 4 ; j++) {
	  cout << cmatrix[i][j] << "\t";
      }
      cout << endl;
  }*/
  return 0;   
}

