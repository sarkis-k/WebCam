import cv2
import time

#var to hold the first frame
first_frame=None

#var for turning camera on
video=cv2.VideoCapture(0)

#while loop for capturing every frame and analyzing and showing
#exit loop hitting 'q'
while True:
	#read every frame: method returns two variables first boolean if frame exists second the actual frame
	check, frame = video.read()

	status=0 	
	
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

	#displaying all frames 
	cv2.imshow("gray", gray)
	cv2.imshow("delta", delta_frame)
	cv2.imshow("thresh", thresh_frame)
	cv2.imshow("color", frame)

	#frame rates are 1 msec
	key = cv2.waitKey(1) 

	#key to eqit the program
	if key==ord('q'):
		break
#destroy all frames
video.release()
cv2.destroyAllWindows()
