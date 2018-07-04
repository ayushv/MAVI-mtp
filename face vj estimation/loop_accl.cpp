#include"loop.h"
#include <ctime>
#include <sys/time.h>
void loop_ac(const int* ptr, size_t step, double begint)
{
	size_t fi; 
	size_t nfeatures = 2*1047; // Total 2094 features
	
//std::cout<< "inside this" << std::endl;
	for (fi = 0; fi < nfeatures; fi++)
	{
#pragma HLS PIPELINE
		my_feature_p1_1[fi] = ptr + my_feature_rect1_Rectx[fi] + (step) * my_feature_rect1_Recty[fi];
    		/* (x + w, y) */                                                      
    		my_feature_p1_2[fi] = ptr + my_feature_rect1_Rectx[fi] + (step) * my_feature_rect1_Recty[fi] + my_feature_rect1_Rectwidth[fi] ;
    		/* (x + w, y) */                                                      
    		my_feature_p1_3[fi] = ptr + my_feature_rect1_Rectx[fi] + (step) * (my_feature_rect1_Recty[fi] + my_feature_rect1_Recthieght[fi]);
    		/* (x + w, y + h) */                                                  
    		my_feature_p1_4[fi] = ptr + my_feature_rect1_Rectx[fi] + (step) * (my_feature_rect1_Recty[fi] + my_feature_rect1_Recthieght[fi]) + my_feature_rect1_Rectwidth[fi];
	}
	for (fi = 0; fi < nfeatures; fi++)
	{
		
#pragma HLS PIPELINE
		my_feature_p2_1[fi] = ptr + my_feature_rect2_Rectx[fi] + (step) * my_feature_rect2_Recty[fi];
    		/* (x + w, y) */                                                      
    		my_feature_p2_2[fi] = ptr + my_feature_rect2_Rectx[fi] + (step) * my_feature_rect2_Recty[fi] + my_feature_rect2_Rectwidth[fi];
    		/* (x + w, y) */                                                      
    		my_feature_p2_3[fi] = ptr + my_feature_rect2_Rectx[fi] + (step) * (my_feature_rect2_Recty[fi] + my_feature_rect2_Recthieght[fi]);
    		/* (x + w, y + h) */                                                  
    		my_feature_p2_4[fi] = ptr + my_feature_rect2_Rectx[fi] + (step) * (my_feature_rect2_Recty[fi] + my_feature_rect2_Recthieght[fi]) + my_feature_rect2_Rectwidth[fi];
	}
	for (fi = 0; fi < nfeatures; fi++)
	{
#pragma HLS PIPELINE
		if (my_feature_rect3_weight[fi])
		{
            		my_feature_p3_1[fi] = ptr + my_feature_rect3_Rectx[fi] + (step) * my_feature_rect3_Recty[fi];
    			/* (x + w, y) */                                                      
    			my_feature_p3_2[fi] = ptr + my_feature_rect3_Rectx[fi] + (step) * my_feature_rect3_Recty[fi] + my_feature_rect3_Rectwidth[fi];
    			/* (x + w, y) */                                                      
    			my_feature_p3_3[fi] = ptr + my_feature_rect3_Rectx[fi] + (step) * (my_feature_rect3_Recty[fi] + my_feature_rect3_Recthieght[fi]);
    			/* (x + w, y + h) */                                                 
    			my_feature_p3_4[fi] = ptr + my_feature_rect3_Rectx[fi] + (step) * (my_feature_rect3_Recty[fi] + my_feature_rect1_Recthieght[fi]) + my_feature_rect3_Rectwidth[fi];
            	
            	}
//std::cout<<((double)getTickCount()-begint)/getTickFrequency()/4.5<<std::endl;

	}
		

}

