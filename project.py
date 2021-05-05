import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]


# You first pass in the image and cascade 
# names as command-line arguments. We’ll use the 
# ABBA image as well as the default cascade for 
# detecting faces provided by OpenCV.

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Now we create the cascade and initialize it with
# our face cascade. This loads the face cascade into 
# memory so it’s ready for use. Remember, the cascade 
# is just an XML file that contains the data to detect faces.

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Here we read the image and convert it to grayscale. 
# Many operations in OpenCV are done in grayscale.

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    # flags = cv2.cv.CV_HAAR_SCALE_IMAGE           #listed as known issue
)

# This function detects the actual face and 
# is the key part of our code, so let’s go over the options:
# 1. The detectMultiScale function is a general function that 
# detects objects. Since we are calling it on the face cascade,
# that’s what it detects.
# 2. The first option is the grayscale image.
# 3. The second is the scaleFactor. Since some faces may be 
# closer to the camera, they would appear bigger than the faces
# in the back. The scale factor compensates for this.
# 4. The detection algorithm uses a moving window to detect 
# objects. minNeighbors defines how many objects are detected 
# near the current one before it declares the face found. minSize, 
# meanwhile, gives the size of each window.
# **Note: I took commonly used values for these fields. In real life, 
# you would experiment with different values for the window size, 
# scale factor, and so on until you found one that works best for you.

# The function returns a list of rectangles in which it believes it found
# a face. Next, we will loop over where it thinks it found something.
print(f"Found {len(faces)} faces!")

# Draw a rectangle around the faces
for x, y, w, h in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


# This function returns 4 values: the x and y location of the rectangle, 
# and the rectangle’s width and height (w , h).

# We use these values to draw a rectangle using the built-in 
# rectangle() function.
cv2.imshow("Faces found", image)
cv2.waitKey(0)





# ****Checking the Results***

# Let’s test against the ABBA photo:
# Enter the following command into the terminal
# $ python project.py abba.png haarcascade_frontalface_default.xml

# Recieved this warning when pushing to repo:
# warning: LF will be replaced by CRLF in haarcascade_frontalface_default.xml.
# The file will have its original line endings in your working directory
