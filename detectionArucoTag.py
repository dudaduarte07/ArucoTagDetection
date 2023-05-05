import numpy as np
import time
import cv2 as cv


ARUCO_DICT = {
	"DICT_4X4_50": cv.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}


def aruco_display(corners, ids, rejected, image):
	if len(corners) > 0:
		
		ids = ids.flatten()
		
		for (markerCorner, markerID) in zip(corners, ids):
			
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			
			cv.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))
			
	return image




aruco_type = "DICT_6X6_100" #setar qual tipo é pra detectar

arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

arucoParams = cv.aruco.DetectorParameters()


cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened():
    
	ret, img = cap.read()

	h, w, _ = img.shape

	width = 1000
	height = int(width*(h/w))
	img = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)
 
	corners, ids, rejected = cv.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)

	detected_markers = aruco_display(corners, ids, rejected, img)

	cv.imshow("Image", detected_markers)

	key = cv.waitKey(1) & 0xFF
	if key == ord("q"):
	    break

cv.destroyAllWindows()
cap.release()