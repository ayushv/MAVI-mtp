#!/usr/bin/python
from openfaceClassifier import *
from animal import *
import glob
import globalt
import subprocess
def face_process():
	print "face_thread"
	inputs=glob.glob('./FR_Train_image/Barun/1/frame1*.jpg')
	arr=['--verbose','infer','classifier.pkl','a.jpg']
	arr+=inputs
	main(arr)
	
	
	
def animal_process():
	print "animal_thread"
	debug = True
	inputs=glob.glob('./test/00004*.jpg')
	objectDetectionProcess(inputs)

def signboard_process():
	print "signboard thread"
	inputs=glob.glob('./sign/truesignboard_DSC0714*.JPG')
	arr=['./SBD','0','1']
	arr+=inputs
	popen=subprocess.Popen(arr,stdout=subprocess.PIPE)
	popen.wait()
	output = popen.stdout.read()
	print output

	
#import thread
from multiprocessing import Process
def coex(inp):
	fp=Process(target=face_process)
	ap=Process(target=animal_process)
	sp=Process(target=signboard_process)
	if(inp=="af"):
		fp.start()
		ap.start()
		fp.join()
		ap.join()
	if(inp=="sf"):
		fp.start()
		sp.start()
		sp.join()
		fp.join()
	if(inp=="sa"):
		ap.start()
		sp.start()
		ap.join()
		sp.join()
	if(inp=="all"):
		ap.start()
		fp.start()
		sp.start()
		ap.join()
		fp.join()
		sp.join()

if(sys.argv[1]=='1'):
	face_process()
elif (sys.argv[1]=='2'):
	animal_process()
elif (sys.argv[1]=='3'):
	signboard_process()
else:
	coex(sys.argv[1])
	
