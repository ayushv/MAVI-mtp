#include<iostream>
#include <cstdio>
#include <string>
#include <map>
#include <deque>
#include <iostream>
#include <cstdlib>
//#include "sds_lib.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

#ifndef _LOOP_H_
#define _LOOP_H_
void calc_pred(float my_feature_rect_weight_featureIdx,int my_feature_p_1_feature_offset,int my_feature_p_2_feature_offset,int my_feature_p_3_feature_offset, int my_feature_p_4_feature_offset, double *A);
void loop_ac(const int* ptr, size_t step,double begint);
int predict(double& sum);
void fr_recog(Mat test_face, int* label);
extern int offset;

extern double varianceNormFactor;

extern int DTreeNode_node_array_featureIdx[2094];
extern float DTreeNode_node_array_threshold[2094];
extern int DTreeNode_node_array_left[2094];
extern int DTreeNode_node_array_right[2094];
extern int DTreeNode_node_array_threshold_fix_pt[2094];

extern int DTree_classifier_array_nodeCount[1047];

extern int Stage_stages_array_first[20];
extern int Stage_stages_array_ntrees[20];
extern float Stage_stages_array_threshold[20];
extern int Stage_stages_array_threshold_fix_pt[20];

extern float leaves_array[3*1047];
extern int leaves_array_fix_pt[3*1047];

extern unsigned int my_feature_rect1_Rectx[2094];
extern unsigned int my_feature_rect1_Recty[2094];
extern unsigned int my_feature_rect1_Rectwidth[2094];
extern unsigned int my_feature_rect1_Recthieght[2094];
extern float my_feature_rect1_weight[2094];
 	// rect[2]
extern unsigned int my_feature_rect2_Rectx[2094];
extern unsigned int my_feature_rect2_Recty[2094];
extern unsigned int my_feature_rect2_Rectwidth[2094];
extern unsigned int my_feature_rect2_Recthieght[2094];
extern float my_feature_rect2_weight[2094];
 	// rect[3]
extern unsigned int my_feature_rect3_Rectx[2094];
extern unsigned int my_feature_rect3_Recty[2094];
extern unsigned int my_feature_rect3_Rectwidth[2094];
extern unsigned int my_feature_rect3_Recthieght[2094];
extern float my_feature_rect3_weight[2094];
 	// p matrix [3][4]
 		//p[1][]
extern const int* my_feature_p1_1[2094];
extern const int* my_feature_p1_2[2094];
extern const int* my_feature_p1_3[2094];
extern const int* my_feature_p1_4[2094];
 		//p[2][]
extern const int* my_feature_p2_1[2094];
extern const int* my_feature_p2_2[2094];
extern const int* my_feature_p2_3[2094];
extern const int* my_feature_p2_4[2094];
 		//p[3][]
extern const int* my_feature_p3_1[2094];
extern const int* my_feature_p3_2[2094];
extern const int* my_feature_p3_3[2094];
extern const int* my_feature_p3_4[2094];
#endif 
