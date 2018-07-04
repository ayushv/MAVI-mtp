//#include "opencv2/videoio.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core_c.h"
//#include "opencv2/core/internal.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cstdio>
#include <string>
#include "opencv2/core/core.hpp"
#include <map>
#include <deque>
#include <iostream>
#include <fstream>
#include <csignal>
#include <ctime>
#include <sys/time.h>
//#define INTERVAL 100

#include "loop.h"

#define CC_CASCADE_PARAMS "cascadeParams"
#define CC_STAGE_TYPE     "stageType"
#define CC_FEATURE_TYPE   "featureType"
#define CC_HEIGHT         "height"
#define CC_WIDTH          "width"

#define CC_STAGE_NUM    "stageNum"
#define CC_STAGES       "stages"
#define CC_STAGE_PARAMS "stageParams"

#define CC_BOOST            "BOOST"
#define CC_MAX_DEPTH        "maxDepth"
#define CC_WEAK_COUNT       "maxWeakCount"
#define CC_STAGE_THRESHOLD  "stageThreshold"
#define CC_WEAK_CLASSIFIERS "weakClassifiers"
#define CC_INTERNAL_NODES   "internalNodes"
#define CC_LEAF_VALUES      "leafValues"

#define CC_FEATURES       "features"
#define CC_FEATURE_PARAMS "featureParams"
#define CC_MAX_CAT_COUNT  "maxCatCount"

#define CC_HAAR   "HAAR"
#define CC_RECTS  "rects"
#define CC_TILTED "tilted"

#define CC_LBP  "LBP"
#define CC_RECT "rect"

#define CV_SUM_PTRS( p0, p1, p2, p3, sum, rect, step )                    \
    /* (x, y) */                                                          \
    (p0) = sum + (rect).x + (step) * (rect).y,                            \
    /* (x + w, y) */                                                      \
    (p1) = sum + (rect).x + (rect).width + (step) * (rect).y,             \
    /* (x + w, y) */                                                      \
    (p2) = sum + (rect).x + (step) * ((rect).y + (rect).height),          \
    /* (x + w, y + h) */                                                  \
    (p3) = sum + (rect).x + (rect).width + (step) * ((rect).y + (rect).height)

#define CALC_SUM_(p0, p1, p2, p3, offset) \
    ((p0)[offset] - (p1)[offset] - (p2)[offset] + (p3)[offset])

#define CALC_SUM(rect,offset) CALC_SUM_((rect)[0], (rect)[1], (rect)[2], (rect)[3], offset)



using namespace cv;
using namespace std;
vector<Rect> faces;
 vector<Rect> candidates;
const double GROUP_EPS = 0.2;
Mat image;
std::string output;
vector<double> percents;
double begint;
double tickf=getTickFrequency();        
class  FeatureEvaluator
{
public:
    enum { HAAR = 0 };
    virtual ~FeatureEvaluator();
    virtual bool read(const FileNode& node);
    virtual int getFeatureType() const;
    static Ptr<FeatureEvaluator> create(int type);
};

//My DS Begin.................................
struct my_feature{
        struct
        {
            Rect r;
            float weight;
        } rect[3];
        const int* p[3][4];
}my_features[2*1047];

Mat sum0,sqsum0,sum1,sqsum;
const int *sump[4];
const double *sqsumpq[4];
int offset;
double varianceNormFactor;
// My Feature structure Structure 
	// rect[1] 
unsigned int my_feature_rect1_Rectx[2094];
unsigned int my_feature_rect1_Recty[2094];
unsigned int my_feature_rect1_Rectwidth[2094];
unsigned int my_feature_rect1_Recthieght[2094];
float my_feature_rect1_weight[2094];
	// rect[2]
unsigned int my_feature_rect2_Rectx[2094];
unsigned int my_feature_rect2_Recty[2094];
unsigned int my_feature_rect2_Rectwidth[2094];
unsigned int my_feature_rect2_Recthieght[2094];
float my_feature_rect2_weight[2094];
	// rect[3]
unsigned int my_feature_rect3_Rectx[2094];
unsigned int my_feature_rect3_Recty[2094];
unsigned int my_feature_rect3_Rectwidth[2094];
unsigned int my_feature_rect3_Recthieght[2094];
float my_feature_rect3_weight[2094];
	// p matrix [3][4]
		//p[1][]
const int* my_feature_p1_1[2094];
const int* my_feature_p1_2[2094];
const int* my_feature_p1_3[2094];
const int* my_feature_p1_4[2094];
		//p[2][]
const int* my_feature_p2_1[2094];
const int* my_feature_p2_2[2094];
const int* my_feature_p2_3[2094];
const int* my_feature_p2_4[2094];
		//p[3][]
const int* my_feature_p3_1[2094];
const int* my_feature_p3_2[2094];
const int* my_feature_p3_3[2094];
const int* my_feature_p3_4[2094];
// Ends here
/*
struct  DTreeNode
        {
            int featureIdx;//ID for the node
            float threshold; // for ordered features only
            int left;//whether have left node 
            int right;
        };
*/

//DTreeNode node_array[2094];
int DTreeNode_node_array_featureIdx[2094];
float DTreeNode_node_array_threshold[2094];
int DTreeNode_node_array_left[2094];
int DTreeNode_node_array_right[2094];
int DTreeNode_node_array_threshold_fix_pt[2094];

/*
struct DTree
        {
            int nodeCount;//no of nodes in the tree
        };
*/

//DTree classifier_array[1047];
int DTree_classifier_array_nodeCount[1047];

/*
struct Stage
        {
            int first;
            int ntrees;
            float threshold;
        };
 */       
//Stage stages_array[20];
int Stage_stages_array_first[20];
int Stage_stages_array_ntrees[20];
float Stage_stages_array_threshold[20];
int Stage_stages_array_threshold_fix_pt[20];

float leaves_array[3*1047];
int leaves_array_fix_pt[3*1047];

Rect normrect;

Size originalWindowSize(20,20);

unsigned int loop_1_t = 0;
unsigned int loop_2_t = 0;
unsigned int loop_3_t = 0;
//--------------My DS end...............


//--------------------------------------------------------------------------------------------------------------------------------------//
//----------Code for group rectangle----------
class SimilarRects
{
public:
    SimilarRects(double _eps) : eps(_eps) {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        double delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;
        return std::abs(r1.x - r2.x) <= delta &&
            std::abs(r1.y - r2.y) <= delta &&
            std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
            std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
    }
    double eps;
};


void groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps)
{
    if( groupThreshold <= 0 || rectList.empty() )
    {
        return;
    }

    vector<int> labels;
    int nclasses = partition(rectList, labels, SimilarRects(eps));

    vector<Rect> rrects(nclasses);
    vector<int> rweights(nclasses, 0);
    vector<int> rejectLevels(nclasses, 0);
    vector<double> rejectWeights(nclasses, DBL_MIN);
    int i, j, nlabels = (int)labels.size();
    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        rweights[cls]++;
    }
    

    for( i = 0; i < nclasses; i++ )
    {
        Rect r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x*s),
             saturate_cast<int>(r.y*s),
             saturate_cast<int>(r.width*s),
             saturate_cast<int>(r.height*s));
    }

    rectList.clear();
    
    for( i = 0; i < nclasses; i++ )
    {
        Rect r1 = rrects[i];
        int n1 =  rweights[i];
        double w1 = rejectWeights[i];
        if( n1 <= groupThreshold )
            continue;
        // filter out small face rectangles inside large rectangles
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];

            if( j == i || n2 <= groupThreshold )
                continue;
            Rect r2 = rrects[j];

            int dx = saturate_cast<int>( r2.width * eps );
            int dy = saturate_cast<int>( r2.height * eps );

            if( i != j &&
                r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }

        if( j == nclasses )
        {
            rectList.push_back(r1);
           
        }
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------//
//---------code for group rectangle finished----------//

class HaarEvaluator : public FeatureEvaluator
{
public:
    struct Feature
    {
        Feature();

        float calc( int offset ) const;
        void updatePtrs( const Mat& sum );
        bool read( const FileNode& node );
        bool tilted;
        enum { RECT_NUM = 3 };
        struct
        {
            Rect r;
            float weight;
        } rect[RECT_NUM];
        const int* p[RECT_NUM][4];
    };

    HaarEvaluator();
    virtual ~HaarEvaluator();
    virtual bool read( const FileNode& node );
    virtual int getFeatureType() const { return FeatureEvaluator::HAAR; }
    double operator()(int featureIdx) const
    { return featuresPtr[featureIdx].calc(offset) * varianceNormFactor; }
    Size origWinSize;
    Ptr<vector<Feature> > features;
    Feature* featuresPtr; // optimization
    bool hasTiltedFeatures;
    Mat sum0, sqsum0, tilted0;
    Mat sum, sqsum, tilted;
    Rect normrect;
    const int *p[4];
    const double *pq[4];
    int offset;
    double varianceNormFactor;
};

inline HaarEvaluator::Feature :: Feature()
{
    tilted = false;
    rect[0].r = rect[1].r = rect[2].r = Rect();
    rect[0].weight = rect[1].weight = rect[2].weight = 0;
    p[0][0] = p[0][1] = p[0][2] = p[0][3] = p[1][0] = p[1][1] = p[1][2] = p[1][3] = p[2][0] = p[2][1] = p[2][2] = p[2][3] = 0;
}

bool HaarEvaluator::read(const FileNode& node)
{
    features->resize(node.size());//no. of features in our case is 2094 for each tree there are 2 coressponding for each node.
    featuresPtr = &(*features)[0];//now this pointer points to the features stored in features-vector named as "features" 
    FileNodeIterator it = node.begin(), it_end = node.end();
    hasTiltedFeatures = false;
    for(int i = 0; it != it_end; ++it, i++)//for each feature ie 2094 times.... 
    {
        if(!featuresPtr[i].read(*it))
            return false;
        else
        {	
        	for(int j=0;j<3;j++){
        	my_features[i].rect[j].r=featuresPtr[i].rect[j].r;
        	my_features[i].rect[j].weight=featuresPtr[i].rect[j].weight;
        	
       	}
        my_feature_rect1_Rectx[i]	=	my_features[i].rect[0].r.x;
	my_feature_rect1_Recty[i]	=	my_features[i].rect[0].r.y;
	my_feature_rect1_Rectwidth[i]	=	my_features[i].rect[0].r.width;
	my_feature_rect1_Recthieght[i]	=	my_features[i].rect[0].r.height;
	my_feature_rect1_weight[i]	=	my_features[i].rect[0].weight;
	my_feature_rect2_Rectx[i]	=	my_features[i].rect[1].r.x;
	my_feature_rect2_Recty[i]	=	my_features[i].rect[1].r.y;
	my_feature_rect2_Rectwidth[i]	=	my_features[i].rect[1].r.width;
	my_feature_rect2_Recthieght[i]	=	my_features[i].rect[1].r.height;
	my_feature_rect2_weight[i]	=	my_features[i].rect[1].weight;
	my_feature_rect3_Rectx[i]	=	my_features[i].rect[2].r.x;
	my_feature_rect3_Recty[i]	=	my_features[i].rect[2].r.y;
	my_feature_rect3_Rectwidth[i]	=	my_features[i].rect[2].r.width;
	my_feature_rect3_Recthieght[i]	=	my_features[i].rect[2].r.height;
	my_feature_rect3_weight[i]	=	my_features[i].rect[2].weight;
        	
        }
    }
    return true;
}

bool HaarEvaluator::Feature :: read( const FileNode& node )
{
    FileNode rnode = node[CC_RECTS];
    FileNodeIterator it = rnode.begin(), it_end = rnode.end();

    int ri;
    for( ri = 0; ri < RECT_NUM; ri++ )
    {
        rect[ri].r = Rect();//this rect is defined as rect[3];
        rect[ri].weight = 0.f;
    }

    for(ri = 0; it != it_end; ++it, ri++)//for each rect in feature do the following....
    {
        FileNodeIterator it2 = (*it).begin();
        it2 >> rect[ri].r.x >> rect[ri].r.y >> rect[ri].r.width >> rect[ri].r.height >> rect[ri].weight;
    }
    return true;
}

HaarEvaluator::HaarEvaluator()
{
    features = new vector<Feature>();
}
HaarEvaluator::~HaarEvaluator()
{
}

Ptr<FeatureEvaluator> FeatureEvaluator::create( int featureType )
{
    return  Ptr<FeatureEvaluator>(new HaarEvaluator);
}

FeatureEvaluator::~FeatureEvaluator() {}
bool FeatureEvaluator::read(const FileNode&) {return true;}
int FeatureEvaluator::getFeatureType() const {return -1;}

class CascadeClassifier
{
public:
    CascadeClassifier();
    CascadeClassifier( const string& filename );
    virtual ~CascadeClassifier();
    bool load( const string& filename );
    virtual bool read( const FileNode& node );
    int getFeatureType() const;
   enum { BOOST = 0 };
   enum { DO_CANNY_PRUNING = 1, SCALE_IMAGE = 2,
           FIND_BIGGEST_OBJECT = 4, DO_ROUGH_SEARCH = 8 };
    class Data
    {
    public:
        struct  DTreeNode
        {
            int featureIdx;//ID for the node
            float threshold; // for ordered features only
            int left;//whether have left node 
            int right;
        };

        struct DTree
        {
            int nodeCount;//no of nodes in the tree
        };

        struct Stage
        {
            int first;
            int ntrees;
            float threshold;
        };
        bool read(const FileNode &node);
        bool isStumpBased;
        int stageType;
        int featureType;
        int ncategories;
        Size origWinSize;
        vector<Stage> stages;
        vector<DTree> classifiers;
        vector<DTreeNode> nodes;
        vector<float> leaves;
        vector<int> subsets;
    };

    Data data;
    Ptr<FeatureEvaluator> featureEvaluator;
};

CascadeClassifier::CascadeClassifier()
{
}

CascadeClassifier::CascadeClassifier(const string& filename)
{
    load(filename);
}

CascadeClassifier::~CascadeClassifier()
{
}

bool CascadeClassifier::load(const string& filename)
{
    data = Data();
    featureEvaluator.release();
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    if( read(fs.getFirstTopLevelNode()) )
        return true;
    fs.release();
}

bool CascadeClassifier::read(const FileNode& root)
{
    if( !data.read(root) )//first cascade read..
        return false;
    // load features
    featureEvaluator = FeatureEvaluator::create(data.featureType);
    FileNode fn = root[CC_FEATURES];
    if( fn.empty() )
        return false;
    return featureEvaluator->read(fn);///then feature read.....
}

bool CascadeClassifier::Data::read(const FileNode &root)
{
    static const float THRESHOLD_EPS = 1e-5f;
    string stageTypeStr = (string)root[CC_STAGE_TYPE];
    stageType = BOOST;
    string featureTypeStr = (string)root[CC_FEATURE_TYPE];
    featureType = FeatureEvaluator::HAAR;
    origWinSize.width = (int)root[CC_WIDTH];
    origWinSize.height = (int)root[CC_HEIGHT];
    // load feature params
    FileNode fn = root[CC_FEATURE_PARAMS];
    if( fn.empty() )
        return false;
    ncategories = fn[CC_MAX_CAT_COUNT];
    int subsetSize = (ncategories + 31)/32,//0 in our case....
    nodeStep = 3 + 1;//( ncategories>0 ? subsetSize : 1 );
    fn = root[CC_STAGES];    // load stages
    stages.reserve(fn.size());
    classifiers.clear();
    nodes.clear();
    FileNodeIterator it = fn.begin(), it_end = fn.end();
    for( int si = 0; it != it_end; si++, ++it )//for each stage....ie 20 times
    {
        FileNode fns = *it;
        Stage stage;//struct Stage{int first; int ntrees;float threshold;};
        stage.threshold = (float)fns[CC_STAGE_THRESHOLD] - THRESHOLD_EPS;
        fns = fns[CC_WEAK_CLASSIFIERS];//same as max weak count....
        stage.ntrees = (int)fns.size();//no of trees in a stage
        stage.first = (int)classifiers.size();//Doubtful....what its use,..0,
        stages.push_back(stage);
        classifiers.reserve(stages[si].first + stages[si].ntrees);//3,
        FileNodeIterator it1 = fns.begin(), it1_end = fns.end();
        for( ; it1 != it1_end; ++it1 ) // weak trees|for each weak classifier tree in a stage 
        {
            FileNode fnw = *it1;//this point to tree ...
            FileNode internalNodes = fnw[CC_INTERNAL_NODES];
            FileNode leafValues = fnw[CC_LEAF_VALUES];
            if( internalNodes.empty() || leafValues.empty() )
                return false;
            DTree tree;
            tree.nodeCount = 2;//(int)internalNodes.size()/nodeStep;
            classifiers.push_back(tree);
            nodes.reserve(nodes.size() + tree.nodeCount);
            leaves.reserve(leaves.size() + leafValues.size());
            if( subsetSize > 0 )//redundant
                subsets.reserve(subsets.size() + tree.nodeCount*subsetSize);
            FileNodeIterator internalNodesIter = internalNodes.begin(), internalNodesEnd = internalNodes.end();
            for( ; internalNodesIter != internalNodesEnd; ) // for each nodes
            {
                DTreeNode node;
                node.left = (int)*internalNodesIter; ++internalNodesIter;
                node.right = (int)*internalNodesIter; ++internalNodesIter;
                node.featureIdx = (int)*internalNodesIter; ++internalNodesIter;
                if( subsetSize > 0 )
                {
                    for( int j = 0; j < subsetSize; j++, ++internalNodesIter )
                        subsets.push_back((int)*internalNodesIter);
                    node.threshold = 0.f;
                }
                else
                    node.threshold = (float)*internalNodesIter; ++internalNodesIter;
                nodes.push_back(node);
            }
            internalNodesIter = leafValues.begin(), internalNodesEnd = leafValues.end();
            for( ; internalNodesIter != internalNodesEnd; ++internalNodesIter ) // leaves
                leaves.push_back((float)*internalNodesIter);
        }
    }
    return true;
}

CascadeClassifier mycascade;
//---------My functions start-------------------//
void copy_stage()
{
	for(int i=0;i<mycascade.data.stages.size();i++){
		Stage_stages_array_first[i]=mycascade.data.stages[i].first;
		Stage_stages_array_ntrees[i]=mycascade.data.stages[i].ntrees;
		Stage_stages_array_threshold[i]=mycascade.data.stages[i].threshold;
	}
}

void copy_classifier()
{
	for(int i=0;i<mycascade.data.classifiers.size();i++)
		DTree_classifier_array_nodeCount[i]=mycascade.data.classifiers[i].nodeCount;
}

void copy_nodes()
{
	for(int i=0;i<mycascade.data.nodes.size();i++){
		DTreeNode_node_array_featureIdx[i]=mycascade.data.nodes[i].featureIdx;
		DTreeNode_node_array_threshold[i]=mycascade.data.nodes[i].threshold;
		DTreeNode_node_array_left[i]=mycascade.data.nodes[i].left;
		DTreeNode_node_array_right[i]=mycascade.data.nodes[i].right;			
	}
}

void copy_leaves()
{
	for(int i=0;i<mycascade.data.leaves.size();i++){
		leaves_array[i]=mycascade.data.leaves[i];
	}
}
//---------my functions end------------------//
//----------code for loading the xml ends..................//
/**
Create 2 matrices sum0 n sqsum0 of size 1st scaled downed image
go on reducing the size of sum and sqsum mat according to the scaled image of each loop.
Algorithm for setImage...It is under 1 for..loop of scale factor.
1. set the #cols and #rows to the size of scaled image +1.
2. Calculate integral of an image and store it in sum and sqsyum
3. 
*/
//--------------------------------------------------- Consider this loop in acceleration ------------------------------------------------------//

bool setImage( const Mat &image, Size _origWinSize )
{
    int rn = image.rows+1, cn = image.cols+1;//this is scaled down image size.....
    Size origWinSize = _origWinSize;
    normrect = Rect(1, 1, origWinSize.width-2, origWinSize.height-2);
    if( sum0.rows < rn || sum0.cols < cn )
    {
        sum0.create(rn, cn, CV_32S);
        sqsum0.create(rn, cn, CV_64F);
    }
    sum1 = Mat(rn, cn, CV_32S, sum0.data);
    sqsum = Mat(rn, cn, CV_64F, sqsum0.data);
    integral(image, sum1, sqsum);
    const int* sdata = (const int*)sum1.data;
    const double* sqdata = (const double*)sqsum.data;
    size_t sumStep = sum1.step/sizeof(sdata[0]);
    size_t sqsumStep = sqsum.step/sizeof(sqdata[0]);
//p[] belongs to featureevaluator and normrect is usually 18X18
    CV_SUM_PTRS( sump[0], sump[1], sump[2], sump[3], sdata, normrect, sumStep );
    CV_SUM_PTRS( sqsumpq[0], sqsumpq[1], sqsumpq[2], sqsumpq[3], sqdata, normrect, sqsumStep );
    size_t fi, nfeatures = 2*1047;
//---------------------------------------------------------------------------------------------------------------------------------//
    const int* ptr = (const int*)sum1.data;  
    size_t step = sum1.step/sizeof(ptr[0]); 
 

   loop_ac(ptr, step,begint); // Accelerated in programmable logic
   return true;
}

bool  setWindow( Point pt )
{
    if( pt.x < 0 || pt.y < 0 ||
        pt.x + 20 >= sum1.cols ||
        pt.y + 20 >= sum1.rows )
        return false;
    size_t pOffset = pt.y * (sum1.step/sizeof(int)) + pt.x;
    size_t pqOffset = pt.y * (sqsum.step/sizeof(double)) + pt.x;
    int valsum = CALC_SUM(sump, pOffset);
    double valsqsum = CALC_SUM(sqsumpq, pqOffset);
    double nf = (double)normrect.area() * valsqsum - (double)valsum * valsum;
    if( nf > 0. )
        nf = sqrt(nf);
    else
        nf = 1.;
    varianceNormFactor = 1./nf;
    offset = (int)pOffset;
    return true;
}

 int call_predict_t = 0;

bool detectSingleScale( const Mat& image,  Size processingRectSize,
                                            int yStep, double scalingFactor, vector<Rect>& candidates )
{
   
    if( !setImage( image,  originalWindowSize ) )
        return false;
           

   Size winSize(cvRound(/*classifier->data.origWinSize.width**/ scalingFactor*20), cvRound(/*classifier->data.origWinSize.height */ scalingFactor*20));//this size is calculated to give back results with respect to original image....
   for( int y = 0; y < processingRectSize.height; y += yStep )//step through the image.....in col direction.
   {
     for( int x = 0; x < processingRectSize.width; x += yStep )//step through the image in row direction.....
     {
		 	

       double gypWeight/* = new double*/;
          int result = -1;
	if( setWindow(Point(x,y)) ){
	 result = predict(gypWeight);
	 	
	}
			//result = predictOrdered (gypWeight);
        if( result > 0 )
          candidates.push_back(Rect(cvRound(x*scalingFactor), cvRound(y*scalingFactor), winSize.width, winSize.height));
        if( result == 0 )
                    x += yStep;
    }
  }
  //	if(call_predict_t%2==0){
double runt= ((double)getTickCount()-begint)/tickf;
//percents.push_back((double)getTickCount()-begint);
double estimate;
	 switch (call_predict_t){
		 case 0:
		  estimate=(100/31.9)*(runt);
          //cout<<"[32/100] Expected Runtime:"<<estimate<<endl;
          break;
         case 1:
			 estimate=(100/53.29)*(runt);
          //cout<<"[54/100] Expected Runtime:"<<estimate<<endl;
          break;
         case 2:
          estimate=(100/68.6)*(runt);
          //cout<<"[68/100] Expected Runtime:"<<estimate<<endl;
          break;
         case 3:
          estimate=(100/79.6)*(runt);
          //cout<<"[79/100] Expected Runtime:"<<estimate<<endl;
          break; 
}	
if(call_predict_t<4){percents.push_back(estimate);			
	 } call_predict_t++;

// cout << " Total #Calls to predict for Single Window : " << call_predict << endl;
// call_predict_t = call_predict_t + call_predict;
    return true;
}
void detectMultiScale( const Mat& image, vector<Rect>& objects,double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize)
{
    
    Mat grayImage = image;
    Size grayImage_sz = image.size();
    if(maxObjectSize.width == 0 || maxObjectSize.height == 0)
	maxObjectSize = grayImage_sz;
    Mat imageBuffer(image.rows + 1, image.cols + 1, CV_8U);
   //change to suitable DS of x,y,Width,hieght..//from cascade file from width and hieght can be replaced by 20 20
    
    for( double factor = 1; ; factor *= scaleFactor )
    {
//This block can be made constant for constant scale factor and image size......begin
        Size windowSize( cvRound(/*originalWindowSize.width**/factor*20), cvRound(/*originalWindowSize.height**/factor*20) );
        Size scaledImageSize( cvRound( grayImage.cols/factor ), cvRound( grayImage.rows/factor ) );//scaling window size...
        Size processingRectSize( scaledImageSize.width - 20, scaledImageSize.height - 20 );

	if(windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height || windowSize.width > grayImage_sz.width ||  windowSize.height > grayImage_sz.height)
	break;
//......end

        if( processingRectSize.width <= 0 || processingRectSize.height <= 0 )
            break;
        if( windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height )
            continue;
        Mat scaledImage( scaledImageSize, CV_8U, imageBuffer.data );
        resize( grayImage, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR );
        int yStep;
        //yStep = factor > 2. ? 1 : 2;
        yStep = 1;//factor > 2. ? 1 : 2;
        
        if( !detectSingleScale( scaledImage, processingRectSize, yStep, factor, candidates ) )
            break;
    }
   // percents.push_back((double)getTickCount()-begint);
 //   cout << "****************************************************************************************" << endl;	
  //  cout << "Total # of calls to Single Detect : " << call_single << endl;
  //  cout << "  Total # calls to predict for all single detect : " << call_predict_t << endl;
  //  cout << "     Total run of loop1 for all cases : " << loop_1_t << endl;
  //  cout << "     Total run of loop2 for all cases : " << loop_2_t << endl;
  //  cout << "     Total run of loop3 for all cases : " << loop_3_t << endl;


    objects.resize(candidates.size());
    std::copy(candidates.begin(), candidates.end(), objects.begin());
    groupRectangles( objects, minNeighbors, GROUP_EPS );
}

Size size1(640,480);
Size size(480,640);


 void signalHandler( int signum ) {
	 cout<<"stoped at : "<<((double)getTickCount()-begint)/tickf<<endl;

   cout << "Interrupt signal (" << signum << ") received.\n";
	cout<<"size: " <<candidates.size()<<endl;
	faces.resize(candidates.size());
    std::copy(candidates.begin(), candidates.end(), faces.begin());
    groupRectangles( faces, 2, GROUP_EPS );
	
	cout<<"size: " <<faces.size()<<endl;
for(size_t i=0;i<faces.size();i++) {
		Rect r = faces[i];
//		string text = /*"Detected Face"*/"FD";
	 rectangle(image, r.tl(), r.br(), cv::Scalar(0,255,0), 2);	}	
imwrite(output, image);

ofstream myfile;
myfile.open ("outcut.csv",ios::app);
myfile<<candidates.size()<<",";
myfile<<faces.size()<<"\n";
myfile.close();

/*
    
   // cleanup and close up stuff here  
   // terminate program  
    
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
*/
   exit(signum);  
}

int main(int argc, const char** argv )
{
  begint = (double)getTickCount();
 signal(SIGINT, signalHandler); 
    signal(SIGALRM, signalHandler); 

int  INTERVAL=(int)(atof(argv[3])*atof(argv[4])*1000);
	 struct itimerval it_val;
	  it_val.it_value.tv_sec =     INTERVAL/1000;
	  it_val.it_value.tv_usec =    (INTERVAL*1000) % 1000000;   
	  it_val.it_interval = it_val.it_value;
	  if (setitimer(ITIMER_REAL, &it_val, NULL) == -1) {
	 	perror("error calling setitimer()");
	    exit(1);
	  }
	string cascadeName;
	double timer_1=0,timer_2=0,timer_3=0; 
	output=argv[2];
	cascadeName = "haarcascade_frontalface_alt2.xml"; // Name of the xml file 
	//cout<< " \nMAVI  : Face Detection and Recognition module (version 1.0)"<<endl;
	//timer_1 = (double)cvGetTickCount();
	
	mycascade.load( cascadeName );
	//VideoCapture vcap(0);
	int count = 0;
//	VideoCapture vcap(argv[1]);
//	      if(!vcap.isOpened()){
//             	cout << "Error opening video stream or file" << endl;
//             	return -1;
//      	      }
//	//VideoWriter video(/*"out.avi"*/argv[2],CV_FOURCC('C','J','P','G'),/*4*/13 , Size(848, 480), 0);
//	VideoWriter video(/*"out.avi"*/argv[2],CV_FOURCC('C','J','P','G'),/*4*/13 , Size(640, 480), 0);
// for(;;) {
	image=imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE); 
	Mat frame;
//	Mat image;
//	vcap >> frame;
	//cvtColor(frame, image, CV_BGR2GRAY);
	//cvtColor(image, image, CV_BGR2GRAY);
	//imshow("Captured GRAY", image);
	//printf("Video : %s\n",argv[1]);
	//printf("Image : %s\n",argv[1]);
	copy_leaves();
	copy_nodes();
	copy_classifier();
	copy_stage();

	if(image.rows>image.cols){
		//image = image(Rect(0, 120, 480, 500));
		resize(image,image,size); 
	}
	else {
	//	image = image(Rect(104, 0, 744, 480));
		//image = image(Rect(240, 0, 1440, 1080));
		resize(image,image,size1);
	}
	                                                                                     
//	mode_fd_fr = atoi(argv[2]);
	//float sf=atof(argv[3]);
	float scale_fac = 1.2; // Selected scale factor is 1.2     
 double time1= (double)getTickCount()-begint;

	
		double estimate=(100/1.9)*(time1/tickf);
		percents.push_back(estimate);
          //cout<<"[2/100] Expected Runtime:"<<estimate<<endl;
          
detectMultiScale(image, faces, scale_fac, 2,0, Size(20, 20), Size(0, 0));
	time1= (double)getTickCount() - begint;
	

	//printf( "detection cycles = %g cycles\n", t2/n );
//	printf( "Detection time = %g ms\n", (timer_2/((double)cvGetTickFrequency()*1000.)));
	cout<<"Face Detection 2.0 #FD : "<<faces.size()<< ", ";
	
	//if(mode_fd_fr == 1){
		for(size_t fd_no=0; fd_no < faces.size(); fd_no++) {	
			int* label = new int;
			string text;
			Rect r = faces[fd_no];
			//cout << (fd_no+1) << ". " << "x: " << r.x << ", y: " << r.y << ", Width: " << r.width << ", Height: " << r.height << ", "; 
			/*if(mode_fd_fr == 1){
				fr_recog(image(r), label);
				if (*label == 0){
					text = "Manish";
                                	cout<< "." << text << ", ";
				}
	                        else if (*label == 1){
					text = "Prof. M. Balakrishnan";
					cout<< "." << text << ", ";
				}
                                else if (*label == 2){
					text = "Sachin";
					cout<< "." << text << ", ";
				}
				else if (*label == 3){
                                        text = "Anupam";
//                                      cout<< *label << "." << text << ", ";
                                        cout<< "." << text << ", ";
                                }
                               else if (*label == 4){
                                       //Mat img=imread("../Recognized_face/4.jpg",CV_LOAD_IMAGE_GRAYSCALE);
                                       //resize(img, img, Size (200, 200));
                                       //imshow("Recognized Person", img);
                                        text = "Dr. Chetan Arora";
//                                      cout<< *label << "." << text << ", ";
                                        cout<< "." << text << ", ";
                                }
                               else if (*label == 5){
                                       //Mat img=imread("../Recognized_face/5.jpg",CV_LOAD_IMAGE_GRAYSCALE);
                                       //resize(img, img, Size (200, 200));
                                       //imshow("Recognized Person", img);
                                        text = "Yoosuf";
//                                      cout<< *label << "." << text << ", ";
                                        cout<< "." << text << ", ";
                                }
				else if (*label == 6){
                                        //Mat img=imread("../Recognized_face/6.jpg",CV_LOAD_IMAGE_GRAYSCALE);
                                        //resize(img, img, Size (200, 200));
                                        //imshow("Recognized Person", img);
                                         text = "Nipun";
//                                       cout<< *label << "." << text << ", ";
                                         cout<< "." << text << ", ";
                                 }
                                else if (*label == 7){
                                        //Mat img=imread("../Recognized_face/7.jpg",CV_LOAD_IMAGE_GRAYSCALE);
                                        //resize(img, img, Size (200, 200));
                                        //imshow("Recognized Person", img);
                                         text = "Basha";
//                                       cout<< *label << "." << text << ", ";
                                         cout<< "." << text << ", ";
                                 }
                                else if (*label == 8){
                                        //Mat img=imread("../Recognized_face/8.jpg",CV_LOAD_IMAGE_GRAYSCALE);
                                        //resize(img, img, Size (200, 200));
                                        //imshow("Recognized Person", img);
                                         text = "Sid";
//                                       cout<< *label << "." << text << ", ";
                                         cout<< "." << text << ", ";
                                 }
                                else if (*label == 9){
                                        //Mat img=imread("../Recognized_face/9.jpg",CV_LOAD_IMAGE_GRAYSCALE);
                                        //resize(img, img, Size (200, 200));
                                        //imshow("Recognized Person", img);
                                         text = "Rajesh";
//                                       cout<< *label << "." << text << ", ";
                                         cout<< "." << text << ", ";
                                 }
                                else
                                {
                                        //cout << "Not in Database we will put it soon" << endl;
                                        text = "Not in Database";
//                                      cout<< *label << "." << text << ", ";
                                        cout<< "." << text << ", ";
                                }
			}
		
			//cout << *label << endl;
			//if (*label == 0){
			//	Mat img=imread("../Recognized_face/0.jpg",CV_LOAD_IMAGE_GRAYSCALE); 
			//	resize(img, img, Size (200, 200));
			//	imshow("Recognized Person", img);
			//	text = "Manish";
			// }
			//else if (*label == 1){
			//	Mat img=imread("../Recognized_face/1.jpg",CV_LOAD_IMAGE_GRAYSCALE); 
			//	resize(img, img, Size (200, 200));
			//	imshow("Recognized Person", img);
			//	 text = "Prof. M. Balakrishnan";
			// }
			//else if (*label == 2){
			//	Mat img=imread("../Recognized_face/2.jpg",CV_LOAD_IMAGE_GRAYSCALE); 
			//	resize(img, img, Size (200, 200));
			//	imshow("Recognized Person", img);
			//	 text = "Sachin";
			// }
			//else if (*label == 3){
			//	Mat img=imread("../Recognized_face/3.jpg",CV_LOAD_IMAGE_GRAYSCALE); 
			//	resize(img, img, Size (200, 200));
			//	imshow("Recognized Person", img);
			//	 text = "Anupam";
			// }
			//else if (*label == 4){
			//	Mat img=imread("../Recognized_face/4.jpg",CV_LOAD_IMAGE_GRAYSCALE); 
			//	resize(img, img, Size (200, 200));
			//	imshow("Recognized Person", img);
			//	 text = "Dr. Chetan Arora";
			// }
			//else if (*label == 5){
			//	Mat img=imread("../Recognized_face/5.jpg",CV_LOAD_IMAGE_GRAYSCALE); 
			//	resize(img, img, Size (200, 200));
			//	imshow("Recognized Person", img);
			//	 text = "Yoosuf";
			// }
			//else if (*label == 6){
			//	Mat img=imread("../Recognized_face/6.jpg",CV_LOAD_IMAGE_GRAYSCALE); 
			//	resize(img, img, Size (200, 200));
			//	imshow("Recognized Person", img);
			//	 text = "Nipun";
			// }
			//else if (*label == 7){
			//	Mat img=imread("../Recognized_face/7.jpg",CV_LOAD_IMAGE_GRAYSCALE); 
			//	resize(img, img, Size (200, 200));
			//	imshow("Recognized Person", img);
			//	 text = "Basha";
			// }
			//else if (*label == 8){
			//	Mat img=imread("../Recognized_face/8.jpg",CV_LOAD_IMAGE_GRAYSCALE); 
			//	resize(img, img, Size (200, 200));
			//	imshow("Recognized Person", img);
			//	 text = "Sid";
			// }
			//else if (*label == 9){
			//	Mat img=imread("../Recognized_face/9.jpg",CV_LOAD_IMAGE_GRAYSCALE); 
			//	resize(img, img, Size (200, 200));
			//	imshow("Recognized Person", img);
			//	 text = "Rajesh";
			// }
			//else {
			//	cout << "Not in Database we will put it soon" << endl;
			//	text = "Not in Database";
			//}
			//imwrite(argv[4], image(r));
			//cvtColor(image, image, CV_GRAY2BGR);
			//rectangle( image, cvPoint(r.x, r.y),
	        	 //      cvPoint((r.x + r.width), (r.y + r.height)),
	        	  //     Scalar( 100, 155, 25 ), 2, 8);
			//putText(image, text, cvPoint(r.x, r.y + r.height), /*fontFace*/ //FONT_ITALIC  , /*fontScale*/ 0.8, Scalar(0, 0, 255), /*thickness*/ 2, 55);
		}
		
	//	namedWindow(argv[1], 1 );
          //      imshow( argv[1], image );	
	//	waitKey(0);	//break;     
		cout << " "<< endl;

	for(size_t i=0;i<faces.size();i++) {
		Rect r = faces[i];
//		string text = /*"Detected Face"*/"FD";
	 rectangle(image, r.tl(), r.br(), cv::Scalar(0,255,0), 2);	}	
//	//	cvtColor(image, image, CV_GRAY2BGR);
//		//std::ostringstream name;
//		//name << "fd_1920x1080_" << count << ".jpg";
//		//count++;
//	
//		//imwrite(name.str(), image(r));
//		rectangle( image, cvPoint(r.x, r.y),
//	       	       cvPoint((r.x + r.width), (r.y + r.height)),
//	       	       Scalar( 255, 255, 255 ), 2, 8);
//		putText(image, text, cvPoint(r.x, r.y + r.height), /*fontFace*/ /*FONT_HERSHEY_SCRIPT_SIMPLEX*/ FONT_ITALIC  , /*fontScale*/ 0.8, Scalar(0, 0, 255), /*thickness*/ 2, 55);
//	}
//	
//	putText(image, argv[2], cvPoint(50, 50)/* Point2f(10,10)*/, FONT_ITALIC  , /*fontScale*/ 0.8, Scalar(0, 0, 0), /*thickness*/ 2, 55);
//	imshow ("Frame", image);
////	waitKey(0);
//	char c = (char)waitKey(1);
//      	 if( c == 27 ) break;
//	video.write(image);
//imwrite(output, image);

double end = (double)getTickCount();
double total=(end-begint);
//cout<<endl<<"total time: "<<total/tickf<<endl; 
 std::ofstream myfile;
 //myfile.open ("estimatedata.csv",ios::app);
 //for (int i = 0; i <percents.size(); ++i)
 //{
     //myfile<<percents[i]<<",";
     
 //}
//myfile<<total/tickf<<",";
 //myfile<<"\n";
 //myfile.close();

myfile.open ("outcut.csv",ios::app);
//myfile<<total/tickf<<",";
myfile<<candidates.size()<<",";
myfile<<faces.size()<<"\n";
myfile.close();


//cout<<"single s madarchood "<<call_predict_t;

	return 0;
}

