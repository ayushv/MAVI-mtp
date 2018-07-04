import cv2
import numpy as np
import os
import tensorflow as tf
import sys
from collections import defaultdict
from time import time
# ## Env setup
sys.path.append("..")

#API Utils
from utils import label_map_util
from utils import visualization_utils as vis_util
debug = True

def objectDetectionProcess(inputs):
	global noOfAnimals, animalArray, directionArray
	graphName = '../ssd_mobilenet_v1_coco_11_06_2017_frozen_inference_graph.pb'
	#Constants
	PATH_TO_CKPT = graphName
	PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
	NUM_CLASSES = 90
	detection_graph = tf.Graph()
	print("Loading graph")
	with detection_graph.as_default():
	  od_graph_def = tf.GraphDef()
	  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
	    serialized_graph = fid.read()
	    od_graph_def.ParseFromString(serialized_graph)
	    tf.import_graph_def(od_graph_def, name='')
	
	
	# ## Loading label map
	# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	with detection_graph.as_default():
	  with tf.Session(graph=detection_graph) as sess:
		i=0
		timearr=[]
		while i<len(inputs):
			#with lock:
			image_np = cv2.imread(inputs[i])
			print("Read image " ,i)
			i+=1
			image_np = cv2.resize(image_np,(640,480))	#Resizing all image sizes to 640x480
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			# Actual detection.
			startTime = time()
			(boxes, scores, classes, num_detections) = sess.run(
			  [boxes, scores, classes, num_detections],
			  feed_dict={image_tensor: image_np_expanded})
			interval=time()-startTime
			if(i>1):
				timearr.append(interval)
			classes = np.array(classes)
			scores = np.array(scores)

			def getBoxesByClass(classNumber):
				classBoxes = boxes[classes == classNumber]
				classScores = scores[classes == classNumber]
				classBoxes = classBoxes[classScores >= 0.5]
				return classBoxes

			def getDirectionFromBox(box):
				xMean = (box[1] + box[3])/2
				if xMean > 0.65:
					direction = 'right'
				elif xMean < 0.35:
					direction = 'left'
				else:
					direction = 'center'
				if debug:
					print("x = ", xMean, "Direction = ", direction)
				return direction

			tempNoOfAnimals = 0
			tempAnimalArray = []
			tempDirectionArray = []

			for box in getBoxesByClass(1):
				tempNoOfAnimals += 1
				tempAnimalArray.append('Person')
				tempDirectionArray.append(getDirectionFromBox(box))

			for box in getBoxesByClass(18):
				tempNoOfAnimals += 1
				tempAnimalArray.append('Dog')
				tempDirectionArray.append(getDirectionFromBox(box))

			for box in getBoxesByClass(21):
				tempNoOfAnimals += 1
				tempAnimalArray.append('Cow')
				tempDirectionArray.append(getDirectionFromBox(box))

			if debug:
				print(tempAnimalArray,tempDirectionArray)
				#print("Scores = ", scores, " on object detection")
				#print("Classes = ", classes)
				print("Time taken : ",interval)
		print sum(timearr)/float(len(timearr))
			#singleMobilePhoneTransaction()
			#sleep(2)
			#for i in range(len(scores)):
			#  if max(scores[i]) > 0.5:
			#  	if debug:
			#  		print(label_map[classes[i]], "detected in",end="")
			#  	box = boxes[i]
			#  	xMean = (box[1] + box[3])/2
			#  	if xMean > 0.65:
			#  		direction = 'right'
			#  	elif xMean < 0.35:
			#  		direction = 'left'
			#  	else:
			#  		direction = 'center'
			#  	if debug:
			#  		print("x = ", xMean, "Direction = ", direction)
			#  	tempNoOfAnimals += 1
			#  	tempAnimalArray.append(label_map[classes[i]])
			#  	tempDirectionArray.append(direction)

			#noOfAnimals = tempNoOfAnimals
			#animalArray = tempAnimalArray
			#directionArray = tempDirectionArray
			#print(classes)
			#print(boxes[0]*im_width,bboxes[1]*im_height,bboxes[2]*im_width,bboxes[3]*im_height)
#objectDetectionProcess()
