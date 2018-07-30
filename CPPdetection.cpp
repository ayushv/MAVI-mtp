#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "hog.cpp"

#include <cstdlib>
#include <csignal>
#include <ctime>
#include <sys/time.h>
//#define INTERVAL 20000

using namespace cv;
using namespace std;
vector<Rect> found, found_filtered;
vector<double> foundWeights;
std::vector<double> times;
std::vector<double> percent;
Size imgsize(640,480);
Mat img;
double runtime=0;
double tickf=getTickFrequency();        
double begin;
std::string output;
int INTERVAL;
class HOGInvoker :
    public ParallelLoopBody
{
public:
    HOGInvoker( const HOGDescriptor* _hog, const Mat& _img,
        double _hitThreshold, const Size& _winStride, const Size& _padding,
        const double* _levelScale, std::vector<Rect> * _vec, Mutex* _mtx,
        std::vector<double>* _weights=0, std::vector<double>* _scales=0 )
    {
        hog = _hog;
        img = _img;
        hitThreshold = _hitThreshold;
        winStride = _winStride;
        padding = _padding;
        levelScale = _levelScale;
        vec = _vec;
        weights = _weights;
        scales = _scales;
        mtx = _mtx;
    }

    void operator()( const Range& range ) const
    {

        int i, i1 = range.start, i2 = range.end;
        double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1+1] : std::max(img.cols, img.rows);
        Size maxSz(cvCeil(img.cols/minScale), cvCeil(img.rows/minScale));
        Mat smallerImgBuf(maxSz, img.type());
        std::vector<Point> locations;
        std::vector<double> hitsWeights;
        double total=0;
        for( i = i1; i < i2; i++ )
        {
            //double bfore = (double)getTickCount();
            double scale = levelScale[i];
            Size sz(cvRound(img.cols/scale), cvRound(img.rows/scale));
            Mat smallerImg(sz, img.type(), smallerImgBuf.ptr());
            if( sz == img.size() )
                smallerImg = Mat(sz, img.type(), img.data, img.step);
            else
                resize(img, smallerImg, sz);

             

            hog->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride, padding);
            Size scaledWinSize = Size(cvRound(hog->winSize.width*scale), cvRound(hog->winSize.height*scale));
            

            mtx->lock();
            for( size_t j = 0; j < locations.size(); j++ )
            {
                vec->push_back(Rect(cvRound(locations[j].x*scale),
                                    cvRound(locations[j].y*scale),
                                    scaledWinSize.width, scaledWinSize.height));
                if (scales)
                    scales->push_back(scale);
            }
            mtx->unlock();

            if (weights && (!hitsWeights.empty()))
            {
                mtx->lock();
                for (size_t j = 0; j < locations.size(); j++)
                    weights->push_back(hitsWeights[j]);
                mtx->unlock();
            }
            double end = (double)getTickCount();
            total=(end-begin)/tickf;
            double estimate;
            if((i+1)%8==0){
                runtime=total;
                percent.push_back(total);
                switch((i+1)/8){
                    case 1:
                        estimate=(100/35.4)*(runtime);
                        cout<<"[37/100] Expected Runtime:"<<estimate<<endl;
                        break;
                    case 2:
                        estimate=(100/57.07)*(runtime);
                        cout<<"[58/100] Expected Runtime:"<<estimate<<endl;
                        break;
                    case 3:
                        estimate=(100/71)*(runtime);
                        cout<<"[72/100] Expected Runtime:"<<estimate<<endl;
                        break;
                    case 4:
                        estimate=(100/81.7)*(runtime);
                        cout<<"[82/100] Expected Runtime:"<<estimate<<endl;
                        break;
                    case 5:
                        estimate=(100/88.4)*(runtime);
                        cout<<"[89/100] Expected Runtime:"<<estimate<<endl;
                        break;
                    case 6:
                        estimate=100/93*(runtime);
                        cout<<"[93/100] Expected Runtime:"<<estimate<<endl;
                        break;
                    case 7:
                        estimate=100/96.2*(runtime);
                        cout<<"[96/100] Expected Runtime:"<<estimate<<endl;
                        break;
                    case 8:
                        estimate=100/97.9*(runtime);
                        cout<<"[98/100] Expected Runtime:"<<estimate<<endl;
                        break;
                    }   
                
                times.push_back(estimate);
                
            }
        }


    }

private:
    const HOGDescriptor* hog;
    Mat img;
    double hitThreshold;
    Size winStride;
    Size padding;
    const double* levelScale;
    std::vector<Rect>* vec;
    std::vector<double>* weights;
    std::vector<double>* scales;
    Mutex* mtx;
};


void clipObjects(Size sz, std::vector<Rect>& objects,
                 std::vector<int>* a, std::vector<double>* b)
{
    size_t i, j = 0, n = objects.size();
    Rect win0 = Rect(0, 0, sz.width, sz.height);
    if(a)
    {
        CV_Assert(a->size() == n);
    }
    if(b)
    {
        CV_Assert(b->size() == n);
    }

    for( i = 0; i < n; i++ )
    {
        Rect r = win0 & objects[i];
        if( !r.empty() )
        {
            objects[j] = r;
            if( i > j )
            {
                if(a) a->at(j) = a->at(i);
                if(b) b->at(j) = b->at(i);
            }
            j++;
        }
    }

    if( j < n )
    {
        objects.resize(j);
        if(a) a->resize(j);
        if(b) b->resize(j);
    }
}




class hognew : public HOGDescriptor{

public:

	hognew(Size win_size=Size(64, 128), Size block_size=Size(16, 16), Size block_stride=Size(8, 8), Size cell_size=Size(8, 8), int nbins=9, double win_sigma=0, double threshold_L2hys=0.2, bool gamma_correction=true, int nlevels=0)
	: HOGDescriptor(win_size,  block_size, block_stride, cell_size, nbins,  win_sigma,  threshold_L2hys, gamma_correction,  nlevels){}


	virtual void detectMultiScale(
    InputArray _img, std::vector<Rect>& foundLocations,
    double hitThreshold, Size winStride, Size padding,
    double scale0, double finalThreshold, bool useMeanshiftGrouping) const;

};



void hognew::detectMultiScale(
    InputArray _img, std::vector<Rect>& foundLocations,
    double hitThreshold, Size winStride, Size padding,
    double scale0, double finalThreshold, bool useMeanshiftGrouping=0) const
{
    //CV_INSTRUMENT_REGION()
    

    double scale = 1.;
    int levels = 0;

    Size imgSize = _img.size();
    std::vector<double> levelScale;
    for( levels = 0; levels < nlevels; levels++ )
    {
        levelScale.push_back(scale);
        if( cvRound(imgSize.width/scale) < winSize.width ||
            cvRound(imgSize.height/scale) < winSize.height ||
                scale0 <= 1 )
            break;
        scale *= scale0;
    }
    levels = std::max(levels, 1);
    levelScale.resize(levels);

    if(winStride == Size())
        winStride = blockStride;

    // CV_OCL_RUN(_img.dims() <= 2 && _img.type() == CV_8UC1 && scale0 > 1 && winStride.width % blockStride.width == 0 &&
    //     winStride.height % blockStride.height == 0 && padding == Size(0,0) && _img.isUMat(),
    //  ocl_detectMultiScale(_img, foundLocations, levelScale, hitThreshold, winStride, finalThreshold, oclSvmDetector,
    //     blockSize, cellSize, nbins, blockStride, winSize, gammaCorrection, L2HysThreshold, (float)getWinSigma(), free_coef, signedGradient));


    // for (int i = 0; i < foundLocations.size(); ++i)
    // {
    // 	cout<<foundLocations[0]<<endl;
    // }
    std::vector<Rect> allCandidates;
    std::vector<double> tempScales;
    std::vector<double> tempWeights;
    std::vector<double> foundScales;


    double bfore = (double)getTickCount();
    
    Mutex mtx;
    Mat img = _img.getMat();
    Range range(0, (int)levelScale.size());
    HOGInvoker invoker(this, img, hitThreshold, winStride, padding, &levelScale[0], &foundLocations, &mtx, &foundWeights, &tempScales);
    parallel_for_(range, invoker);
    
        /* code */
    
     double end = (double)getTickCount();

    //cout<<"hog end:"<<(end-bfore)/getTickFrequency()<<endl;


    std::copy(tempScales.begin(), tempScales.end(), back_inserter(foundScales));
    // foundLocations.clear();
    //std::copy(allCandidates.begin(), allCandidates.end(), back_inserter(foundLocations));
    //foundWeights.clear();
    //std::copy(tempWeights.begin(), tempWeights.end(), back_inserter(foundWeights));
    
    

    if ( useMeanshiftGrouping )
        groupRectangles_meanshift(foundLocations, foundWeights, foundScales, finalThreshold, winSize);
    else
        groupRectangles(foundLocations, foundWeights, (int)finalThreshold, 0.2);

    // for (int i = 0; i < found.size(); ++i)
    // {
    //     cout<<found[0]<<endl;


    // }

    clipObjects(imgSize, foundLocations, 0, &foundWeights);
   

}





hognew hog;







void signalHandler( int signum ) {
   cout << "Interrupt signal (" << signum << ") received.\n";

    
   // cleanup and close up stuff here  
   // terminate program  
    hog.groupRectangles(found, foundWeights, 2, 0.2);
    clipObjects(imgsize, found, 0, &foundWeights);
   cout<<"found :"<<found.size()<<endl; 
    size_t i, j;
    for (i=0; i<found.size(); i++)
        {
        Rect r = found[i];
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.06);
        r.height = cvRound(r.height*0.9);
        rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
        cout<<  r.tl()<<r.br() << endl; 
       
    }
 imwrite(output,img);
  waitKey(0);
ofstream myfile;
myfile.open ("foundata1.csv",ios::app);
myfile<<found.size()<<"\n";
myfile.close();

   exit(signum);  
}


int main(int argc, char *argv[])
{
         begin = (double)getTickCount();

    if (argc == 1)
		{
			printf("Need input image file\n");
			return 0;
		}

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
	


    img = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file
	output=argv[2];
	hog.load("cvHOGClassifier9.yaml");

    if(! img.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the img" << std::endl ;
        return -1;
    }
	
	int thresh = 2;
	double scale = 1.03;
	int winSize = 16;
	
	Size size(640,480);
	resize(img, img, size);
	double bfore = (double)getTickCount();
    runtime=(bfore-begin)/tickf;
    cout<<"[4/100] Expected Runtime:"<<runtime/0.0414<<endl;
    percent.push_back(runtime);
    times.push_back(runtime/0.0414);
        hog.detectMultiScale(img, found, 0, Size(winSize,winSize), Size(32,32), scale, thresh);
 

 double after = (double)getTickCount();
	cout <<"[99/100] Expected Runtime:"<<runtime/0.988<<endl;
    cout<<"found :"<<found.size()<<endl;
        size_t i, j;
        for (i=0; i<found.size(); i++)
        {
            Rect r = found[i];
            for (j=0; j<found.size(); j++)
                if (j!=i && (r & found[j])==r)
                    break;
            if (j==found.size())
                found_filtered.push_back(r);
        }
        for (i=0; i<found_filtered.size(); i++)
        {
	    Rect r = found_filtered[i];
            r.x += cvRound(r.width*0.1);
	    r.width = cvRound(r.width*0.8);
	    r.y += cvRound(r.height*0.06);
	    r.height = cvRound(r.height*0.9);
	    rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
	    cout<<  r.tl()<<r.br() << endl; 
	   
	}
 imwrite(argv[2],img);
 //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
 //imshow( "Display window", img ); // Show our image inside it.
 waitKey(0); // Wait for a keystroke in the window
    double end = (double)getTickCount();
double total=(end-begin)/tickf;
double intro=(bfore-begin)/tickf;
double main=(after-bfore)/tickf;
double outro=(end-after)/tickf;
// cout<<"intro "<<intro/total<<endl;
// cout<<"main  "<<main/total<<endl;
// cout<<"outro  "<<outro/total<<endl;
cout<<"total: "<<total<<endl;
//double sum=0;
// ofstream myfile;
// myfile.open ("percentdata.csv",ios::app);
// for (int i = 0; i < percent.size(); ++i)
// {
//     percent[i]=percent[i]*100/total;
//     myfile<<percent[i]<<",";
// }
// myfile<<"\n"<<endl;
// myfile.close();


ofstream myfile;
myfile.open ("foundata.csv",ios::app);

myfile<<found.size()<<"\n";
myfile.close();






//        return 0;
}
