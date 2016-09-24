from __future__ import print_function
from facedetector import FaceDetector
import cv2
import os
import numpy as np

# camera = cv2.VideoCapture(0)
# while True:
# 	(grabbed, frame) = camera.read()
#
# 	if (not grabbed):
# 		break
#
# 	# frame = imutils.resize(frame, width = 300)
#
# 	frameClone = frame.copy()
#
# 	gray = cv2.cvtColor(frameClone, cv2.COLOR_BGR2GRAY)
# 	fd = FaceDetector('haarcascade_frontalface_default.xml')
# 	faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,
# 		minSize = (30, 30))
#
# 	for (x, y, w, h) in faceRects:
# 		cv2.rectangle(frameClone, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# 	cv2.imshow("Face", frameClone)
#
# 	waitResult = cv2.waitKey(1)
# 	if waitResult & 0xFF == ord("1") or waitResult & 0xFF == ord("2") or waitResult & 0xFF == ord("3") or waitResult & 0xFF == ord("4") or waitResult & 0xFF == ord("5"):
# 		cv2.imwrite("face"+str(waitResult)+".jpg", frame[faceRects[0][1]:faceRects[0][1]+faceRects[0][3], faceRects[0][0]:faceRects[0][0]+faceRects[0][2]])
#
# 	if waitResult & 0xFF == ord("q"):
# 		break
#
# camera.release()
# cv2.destroyAllWindows()

image = cv2.imread('./yalefaces/subject01.glasses')
cv2.imshow('xx', image)
cv2.waitKey(0)
exit()
# path = './yalefaces'
# fd = FaceDetector('haarcascade_frontalface_default.xml')
# recognizer = cv2.face.createLBPHFaceRecognizer()
# image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
# labels = []
# images = []
# for image_path in image_paths:
# 	print("learn {}".format(image_path))
# 	nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
# 	image = cv2.imread(image_path)
# 	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# 	faces = fd.detect(image, scaleFactor = 1.2, minNeighbors = 5,
# 		minSize = (20, 20))
# 	print("  face {}".format(len(faces)))
# 	for (x, y, w, h) in faces:
# 		images.append(image[y:y+h, x:x+w])
# 		labels.append(nbr)
#
# cv2.imshow('img', image)
#
# if len(labels) > 0:
# 	recognizer.train(images, np.array(labels))
# else:
# 	print("No face")
# 	exit()

# camera = cv2.VideoCapture(0)
# while True:
# 	(grabbed, frame) = camera.read()
#
# 	if (not grabbed):
# 		break
#
# 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	fd = FaceDetector('haarcascade_frontalface_default.xml')
# 	faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,
# 		minSize = (30, 30))
#
#
# 	for (x, y, w, h) in faceRects:
# 		predictImage = gray[y:y+h, x:x+w]
# 		nbrPredicted = recognizer.predict(predictImage)
# 		print("Result is {}".format(nbrPredicted))
#
# 	cv2.imshow("Face", frame)
#
# 	waitResult = cv2.waitKey(1)
# 	if waitResult & 0xFF == ord("q"):
# 		break
#
# camera.release()
# cv2.destroyAllWindows()


frame = cv2.imread('src.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
fd = FaceDetector('haarcascade_frontalface_default.xml')
faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,
	minSize = (30, 30))


for (x, y, w, h) in faceRects:
	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	predictImage = gray[y:y+h, x:x+w]
	nbrPredicted = recognizer.predict(predictImage)
	print("Result is {}".format(nbrPredicted))

cv2.imshow("Face", frame)

cv2.waitKey(0)
	# break

# camera.release()
