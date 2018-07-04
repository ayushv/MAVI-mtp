# MAVI-mtp

For runtime estimation and preemption :  
After installing opencv dependencies just run ./facevj input.jpg output.jpg timetointerruptinseconds

For coexecution :
-setup openface 
-setup tensorflow and get models mobilenetssd
-setup input in coex.py
run python coex.py switch
switch : 
1=face 
2=animal
3-signboard
sa=sign+animal
sf=sign+face
af=animal+face
all=all together
