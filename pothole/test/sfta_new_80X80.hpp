#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

/*-----------------------------------------------------------Otsu Thresholding------------------------------------------------------*/
  double
Otsu( Mat &counts )
{

  int N;
  int sum;
  sum=0;
  N= counts.rows;
  for (int i=0; i<N; i++)
  {
    sum += counts.at<int>(i, 0);
  }
  //cout<<sum<<endl;
  //getchar();
  double mu = 0, scale = 1./sum;

  for(int i = 0; i < N; i++ )
    mu += i*(double)counts.at<int>(i, 0);

  mu *= scale;
  double mu1 = 0, q1 = 0;
  double max_sigma = 0, max_val = 0;

  for(int i = 0; i < N; i++ )
  {
    double p_i, q2, mu2, sigma;

    p_i = counts.at<int>(i, 0)*scale;
    mu1 *= q1;
    q1 += p_i;
    q2 = 1. - q1;

    if( std::min(q1,q2) < FLT_EPSILON || std::max(q1,q2) > 1. - FLT_EPSILON )
      continue;

    mu1 = (mu1 + i*p_i)/q1;
    mu2 = (mu - q1*mu1)/q2;
    sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
    if( sigma > max_sigma )
    {
      max_sigma = sigma;
      max_val = i;
    }
  }

  return max_val;
}
/*----------------------------------------------------------Histogram Function----------------------------------------------------------*/
static int Hist( const Mat& _src, int* h )
{
  Size size = _src.size();
  int step = (int) _src.step;
  if( _src.isContinuous() )
  {
    size.width *= size.height;
    size.height = 1;
    step = size.width;
  }
  const int N = 256;
  int i, j; h[N]={0};
  for( i = 0; i < size.height; i++ )
  {
    const uchar* src = _src.ptr() + step*i;
    j = 0;
#if CV_ENABLE_UNROLLED
    for( ; j <= size.width - 4; j += 4 )
    {
      int v0 = src[j], v1 = src[j+1];
      h[v0]++; h[v1]++;

      v0 = src[j+2]; v1 = src[j+3];
      h[v0]++; h[v1]++;
    }
#endif
    for( ; j < size.width; j++ )
      h[src[j]]++;
  }
  return 0;
}


/*-------------------------------------------------------findBorders------------------------------------------------------------------*/
  static Mat
findBorders( Mat& image )
{

  int rows = image.rows;
  int cols = image.cols;
  bool bkgFound;
  Mat Im, Imm;
  Imm.create(rows,cols, CV_8UC1);
  Imm = Scalar::all(0);

  Im.create(rows+2,cols+2,  CV_8UC1);
  Im = Scalar::all(255);
  Mat ImRoi(Im, Rect(1, 1, cols, rows));
  image.copyTo(ImRoi);
  //cout<<Im<<endl;
  //imshow("aa",Im);
  //waitKey(0);
  //cout<<rows<<endl;
  //cout<<cols<<endl;

  bkgFound = false;
  for (int r =0; r<rows ; r++)
  {
    for (int c =0; c<cols ; c++)
    {
      if (Im.at<uchar>(r+1, c+1)==255)
      {		
	bkgFound = false;

	for (int i =0; i<=2 ; i++)
	{
	  for (int j =0; j<=2 ; j++)
	  {
	    if (Im.at<uchar>(r+i, c+j)==0)
	    {		
	      Imm.at<uchar>(r, c)=255;
	      bkgFound = true;
	      break;
	    }
	  }

	  if (bkgFound==true)
	  {
	    break;
	  }
	}
      }
    }
  }
  //imshow( "Findb", Imm );                   // Show our image inside it.
  //waitKey(0);                                          // Wait for a keystroke in the window
  return Imm;
}

/*------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------Polyfit-------------------------------------------------------------*/
static int polyfit(const Mat& src_x, const Mat& src_y, Mat& dst, int order)
{
  CV_Assert((src_x.rows>0)&&(src_y.rows>0)&&(src_x.cols==1)&&(src_y.cols==1)
      &&(dst.cols==1)&&(dst.rows==(order+1))&&(order>=1));
  Mat X;
  X = Mat::zeros(src_x.rows, order+1,CV_32FC1);
  Mat copy;
  for(int i = 0; i <=order;i++)
  {
    copy = src_x.clone();
    pow(copy,i,copy);
    Mat M1 = X.col(i);
    copy.col(0).copyTo(M1);
  }
  Mat X_t, X_inv;
  transpose(X,X_t);
  Mat temp = X_t*X;
  Mat temp2;
  invert (temp,temp2);
  Mat temp3 = temp2*X_t;
  Mat W = temp3*src_y;
  W.copyTo(dst);
  return 0;
}
/*-------------------------------------------------------------------------------------------------------------------------------*/

/*----------------------------------------------------------------hausDim-------------------------------------------------------------*/
  static double
hausDim( Mat& image )
{
  Mat IPad;  
  int rows = image.rows;
  int cols = image.cols;
  int maxDim = max(rows,cols);
  int newDimSize = pow(2,ceil(log(maxDim)/log(2)));

  IPad.create(newDimSize,newDimSize,  CV_8UC1);
  IPad = Scalar::all(0);
  Mat IPadRoi(IPad, Rect(0, 0, cols, rows));
  image.copyTo(IPadRoi);
  Mat boxCounts,resolutions,D;
  boxCounts.create(8, 1, CV_32FC1);
  resolutions.create(8, 1, CV_32FC1);
  D.create(2, 1, CV_32FC1);
  int iSize, boxSize, boxesPerDim;
  int idx;
  bool objFound;
  int boxCount;
  int minBox[256]={0},maxBox[256]={0};

  iSize=newDimSize;
  boxSize=iSize;
  boxesPerDim=1;
  idx=-1;
  boxCount = 0;
  while (boxSize >= 1) 


  {
    for (int i=0; i<= (iSize-boxSize); i=i+boxSize)
    {
      minBox[i/boxSize]=i;
      //cout<<minBox[i/boxSize]<<"  ";

    }
    /*
       for(int k=0;k<300;k++)
       {
       cout<<minBox[k]<<"  ";
       }
     */

    for (int j=0; j<= (iSize-boxSize); j=j+boxSize)
    {
      maxBox[j/boxSize]=j+boxSize;
      //cout<<maxBox[j/boxSize]<<endl;

    }

    /*
       for(int k=0;k<300;k++)
       {
       cout<<maxBox[k]<<"  ";
       }*/

    for (int boxRow=0; boxRow< boxesPerDim; boxRow++)
    {
      for (int boxCol=0; boxCol< boxesPerDim; boxCol++)
      {
	objFound=false;
	for (int row=minBox[boxRow]; row< maxBox[boxRow]; row++)
	{
	  for (int col=minBox[boxCol]; col< maxBox[boxCol]; col++)
	  {
	    if (IPad.at<uchar>(row, col)==255)
	    {		
	      boxCount=boxCount+1;
	      objFound = true;
	      break;		
	    }
	  }			
	  if (objFound==true)
	  {
	    break;
	  }

	}
      }
    }


    idx=idx+1;
    boxCounts.at<float>(idx)=(float)boxCount;

    resolutions.at<float>(idx)=(float)(1.0/boxSize);
    boxesPerDim =  boxesPerDim*2;
    boxSize = boxSize/2;

    boxCount = 0;
  }
  Mat A,B,C,E;
  Vec4f F;
  double slope;

  log(resolutions,A);
  log(boxCounts,B);

  polyfit(A, B, D, 1);
  slope = D.at<float>(1, 0);
  //cout<<boxCounts<<endl;
  //cout<<"slop="<<slope<<endl;

  //getchar();
  return slope;
}
/*----------------------------------------------------otsu_help---------------------------------------------------------------------*/ 
int otsu_help(int lowerBin, int upperBin, int tLower, int tUpper, int *Threshold, Mat hist)
{

  Mat counts;
  double oth,otsuTh[4];
  int pp,level[4];
  double p;


  if ((tUpper < tLower) || (lowerBin >= upperBin))
    return 0;

  else
  {

    Rect myROI(0, lowerBin, 1, upperBin-lowerBin);
    counts = hist(myROI);

    p = ceil((double)(tLower+tUpper)/2);
    pp =  p;	
    oth = Otsu(counts);
    level[pp] = oth + lowerBin+2;
    otsuTh[pp] =level[pp];
    Threshold[pp]=otsuTh[pp];
    otsu_help(lowerBin, level[pp], tLower, pp-1,Threshold, hist);
    otsu_help(level[pp]+1, upperBin, pp+1, tUpper, Threshold, hist);
  }

  return 0;
}
/*--------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------otsurec----------------------------------------------------------------*/
int otsurec(const Mat& src, int* Threshold, Mat hist)
{
  otsu_help(0, 256, 0, 3, Threshold, hist); 
  return 0;
}
/*--------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------sfta-------------------------------------------------------------------------*/

int sfta( Mat& image,double* D )
{
  int Thresh[5];
  Mat hist;
  int histogram[256]={0};
  Hist(image, histogram);
  hist = Mat(256, 1, CV_32SC1, histogram);
  otsurec(image, Thresh, hist);


  //FindBorder 
  Mat Ib, Ib1,Ib2,image1,image2;
  int thresh, count,v,pos;
  //double D[24] = {0}, sum; 
  double sum,xx; 
  int W1=80;
  int W2=80;
  Mat W;
  pos=0;
  for (int k=0; k<4; k++)
  {
    count=0;
    sum=0;
    thresh = Thresh[k];
    //cout<<thresh<<endl;
    //getchar();
    Ib1 = image > thresh;
    Ib1 = findBorders(Ib1);

    //cout<<Ib1<<endl;
    //imshow("bod1",Ib1);
    //waitKey(0);	

    for (int i=0; i<W2; i++)
    {
      for (int j=0; j<W1; j++)
      {
	if (Ib1.at<uchar>(i, j)==255)
	{
	  v=image.at<uchar>(i, j);
	  sum += v;
	  count = count+1;
	};
      };
    };
    D[pos]= hausDim(Ib1);
    pos=pos + 1;
    D[pos]= sum/count;
    pos=pos + 1;
    D[pos]= count;
    pos=pos + 1;
  }
  //Thresh[4]=225;

  int thresh1, thresh2;
  for (int k=0; k<4; k++)
  {
    Ib2=Ib1;	
    count=0;
    sum=0;
    thresh1 = Thresh[k];
    thresh2 = Thresh[k+1];
    //Ib2 = image > thresh1 ;//&  image < thresh2;
    Ib2 = findBorders(Ib2);	
    //imshow("bnod",Ib2);
    //waitKey(0);	

    for (int i=0; i<W2; i++)
    {
      for (int j=0; j<W1; j++)
      {
	if (Ib2.at<uchar>(i, j)==255)
	{
	  v=image.at<uchar>(i, j);
	  sum += v;
	  count = count+1;
	};
      };
    };
    D[pos]= hausDim(Ib2);
    pos=pos + 1;
    D[pos]= sum/count;
    pos=pos + 1;
    //cout<<count<<endl;	

    D[pos]= count;	
    pos=pos + 1;
  }


  return 0;
}
