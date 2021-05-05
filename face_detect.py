# Grab the face_detect.py script, the abba.png pic, and 
# the haarcascade_frontalface_default.xml 
# from https://github.com/shantnu/FaceDetect/

# Recieved this warning when pushing to repo:
# warning: LF will be replaced by CRLF in haarcascade_frontalface_default.xml.
# The file will have its original line endings in your working directory


import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    # flags = cv2.cv.CV_HAAR_SCALE_IMAGE            #listed as known issue
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)




# ****Checking the Results***

# Let’s test against the ABBA photo:
# Enter the following command into the terminal
# $ python face_detect.py abba.png haarcascade_frontalface_default.xml