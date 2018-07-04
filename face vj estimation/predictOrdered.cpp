#include "loop.h"

using namespace std;
int predict(double& sum)
{
    int nstages = 20;
    int nodeOfs = 0, leafOfs = 0;
   for( int si = 0; si < nstages; si++ )//loop for each stage.....
    {
        
        int wi, ntrees = Stage_stages_array_ntrees[si];
        sum = 0;
        for( wi = 0; wi < ntrees; wi++ )//for each tree in a stage.....
        {
            int idx = 0, root = nodeOfs;
            do
            {
                int featureIdx=DTreeNode_node_array_featureIdx[root + idx];
                double val = 0;//calls operator of Haar Evaluator.....
                double ret = my_feature_rect1_weight[featureIdx] * (*(my_feature_p1_1[featureIdx] + offset) - *(my_feature_p1_2[featureIdx] + offset) - *(my_feature_p1_3[featureIdx] + offset) + *(my_feature_p1_4[featureIdx] + offset)) + my_feature_rect2_weight[featureIdx] * (*(my_feature_p2_1[featureIdx] + offset) - *(my_feature_p2_2[featureIdx] + offset) - *(my_feature_p2_3[featureIdx] + offset) + *(my_feature_p2_4[featureIdx] + offset));
		        if( my_feature_rect3_weight[featureIdx]!= 0.0f )
			        ret +=	my_feature_rect3_weight[featureIdx] * (*(my_feature_p3_1[featureIdx] + offset) - *(my_feature_p3_2[featureIdx] + offset) - *(my_feature_p3_3[featureIdx] + offset) + *(my_feature_p3_4[featureIdx] + offset));
		
                val=ret*varianceNormFactor;
                idx = val < DTreeNode_node_array_threshold[root + idx] ? DTreeNode_node_array_left[root + idx] : DTreeNode_node_array_right[root + idx];
            }
            while( idx > 0 );
            sum += leaves_array[leafOfs - idx];
            nodeOfs += DTree_classifier_array_nodeCount[Stage_stages_array_first[si] + wi];
            leafOfs += DTree_classifier_array_nodeCount[Stage_stages_array_first[si] + wi] + 1;
        }
       if( sum < Stage_stages_array_threshold[si] )
            return -si;
    }
    return 1;
}
