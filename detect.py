import cv2
import time
from datetime import datetime
import pandas as pd

#var to hold the first frame
first_frame=None
status_list=[None, None]					#chronological list of all frames with indication of movement
times=[]									#record of date/time when movement apeard
df=pd.DataFrame(columns=["Start","End"])	#pandas dataframe to organize date/time records


#var for turning camera on
video=cv2.VideoCapture(0)

#while loop for capturing every frame and analyzing and showing
#exit loop hitting 'q'
while True:
	#read every frame: method returns two variables first boolean if frame exists second the actual frame
	check, frame = video.read()

	status=0 		#var for the frame where motion was detected
	
	#change the color of the frame to gray cheme
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#blur the gray frame using gaussian algorithm
	gray = cv2.GaussianBlur(gray, (21,21),0)

	#assigning the very first frame to the var first_frame
	if first_frame is None:
		first_frame=gray
		continue

	#assigning difference between first frame and other frames to a var delta_frame	
	delta_frame=cv2.absdiff(first_frame,gray)
	#clearing delta_frame from unnecessary objects and keeping objects that has obvious movement in the frame, threshold >30
	thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
	thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)
	#method to find contours for all moving objects
	(cnts,_)=cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	#creating green rectangle around the moving object
	for contoure in cnts:
		if cv2.contourArea(contoure)<10000:
			continue
		status=1
		(x,y,w,h)=cv2.boundingRect(contoure)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
	
	status_list.append(status)

	#records the time and date of apeared motion in the frame
	#checks last two items in the status_list if they differ(indicating something changed) records the time
	if status_list[-1]==1 and status_list[-2]==0:
		times.append(datetime.now())
	if status_list[-1]==0 and status_list[-2]==1:
		times.append(datetime.now())



	#displaying all frames 
	cv2.imshow("gray", gray)
	cv2.imshow("delta", delta_frame)
	cv2.imshow("thresh", thresh_frame)
	cv2.imshow("color", frame)

	#frame rates are 1 msec
	key = cv2.waitKey(1) 

	#key to eqit the program
	if key==ord('q'):
		#if quitied while object in the frame record exit time
		if status==1:
			times.append(datetime.now())
		break



print(status_list)
print(times)

#creating pandas dataframe for date/time list
#loop iteration step = 2 for start/end tuple
for i in range(0, len(times), 2):
	df=df.append({"Start":times[i], "End":times[i+1]}, ignore_index=True)

#converting pandas datafrmae to .csv file
df.to_csv("Times.csv")

#destroy all frames
video.release()
cv2.destroyAllWindows()
