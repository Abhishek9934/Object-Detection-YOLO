import numpy as np
import argparse
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True, help ="path to input image")
ap.add_argument("-o","--output" ,required = True ,help ="path to output image" )
args = vars(ap.parse_args())


labels ="yolo/yolov3.txt"
LABELS= open(labels).read().strip().split("\n")
# print(LABELS)

#### list of random colors
np.random.seed(42)
colors =  np.random.randint(0,255,size=(len(LABELS),3),dtype ="uint8")

weightsPath="yolo/yolov3.weights"
configPath = "yolo/yolov3.cfg"

print ("INFO running the model from disk....")
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)

#loading the input image
image= cv2.imread(args["image"])
(h,w) = image.shape[:2]

#get the output layers from the  yolo model

ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]


blob = cv2.dnn.blobFromImage(image , 1/255.0 ,(416,416) , swapRB =True ,crop = False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

print("INFO yolo took {:.6f}seconds".format(end-start))

boxes=[]
confidences =[]
classids=[]

for output in layerOutputs:
	for detection in output:
		#print(detection)
		scores = detection[5:]
		classid = np.argmax(scores)
		confidence = scores[classid]

		if confidence > 0.5:
			box = detection[0:4]*np.array([w,h,w,h])
			(cx,cy,width,height) = box.astype("int")
			x=int(cx-(width/2))
			y= int(cy -(height/2))

			boxes.append([x,y,int(width),int(height)])
			confidences.append(float(confidence))
			classids.append(classid)

idxs = cv2.dnn.NMSBoxes(boxes,confidences,0.5, 0.3)

if len(idxs) >0:
	for i in idxs.flatten():
		(x,y)= (boxes[i][0],boxes[i][1])
		(W,H)=(boxes[i][2],boxes[i][3])

		color = [int(c) for  c in colors[classids[i]]]
		cv2.rectangle(image ,(x,y),(x+W,y+H),color,2)

		text = "{}:{:.4f}".format(LABELS[classids[i]],confidences[i])
		cv2.putText(image,text ,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2)

cv2.imshow("Image",image)
cv2.waitKey()
output_path="output/"+args["output"]
cv2.imwrite(output_path,image)
cv2.destroyAllWindows()

