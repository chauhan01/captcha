import argparse
import cv2
import numpy as np
import os
import glob
from keras.models import load_model
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image directory")
ap.add_argument("-o", "--output", required=True,
	help="path to output image")
ap.add_argument("-m", "--model", required=True,
	help="path to model.hdf5 file")
ap.add_argument("-l", "--labels", required=True,
	help="path to captcha_labels.pickle file file")
args = vars(ap.parse_args())

test_image = args['input']
OUTPUT_FOLDER = args['output']
LB = args['labels']
MODEL = args['model']


#Loading the model
model = load_model(MODEL)

#load labels file
lb = open(LB, "rb")
lb = pickle.load(lb)


# Load the image and convert it to grayscale
image = cv2.imread(test_image, cv2.IMREAD_COLOR)

# Add some extra padding around the image
image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))

#converting to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#applying blur
median = cv2.medianBlur(gray, 1)

# applying threshold
thresh = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]

# applying erode
kernel = np.ones((2, 2), np.uint8)
thresh = cv2.erode(thresh, kernel, iterations=1)

# find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
# connectedComponentswithStats yields every seperated component with information on each of them, such as size
# the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]
nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
# here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 150

# clean image
img = np.zeros((output.shape))
# for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img[output == i + 1] = 255
img = np.uint8(img)

# finding the contours
contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sorting contours
cnts = sorted(contours, key=cv2.contourArea, reverse=True)
boundingBoxes = [cv2.boundingRect(c) for c in cnts]
(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][0], reverse=False))

# creating empty list for holding the coordinates of the letters
letter_image_regions = []  # for letter images
letter_bounding_rect = []  # for letter bounding box

for contour in cnts:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    (x, y, w, h) = cv2.boundingRect(contour)

    letter_image_regions.append(box)
    letter_bounding_rect.append((x, y, w, h))

# Create an output image and a list to hold our predicted letters
output = image.copy()

# Creating an empty list for storing predicted letters
predictions = []

# Save each letter as a single image
for letter_image_region, letter_bounding_box in zip(letter_image_regions, letter_bounding_rect):

    #extracting rotated letter's coordinates
    pts = np.array(letter_image_region, np.int32)

    # creating the mask of rotated rectangle
    mask = np.zeros((img.shape[0], img.shape[1]))

    cv2.fillConvexPoly(mask, pts, 1)
    mask = mask.astype(np.bool)

    # applying mask to the blank image
    out = np.zeros_like(img, np.uint8())
    out[mask] = img[mask]

    #extracting bounding box coordinates
    x, y, w, h = letter_bounding_box

    # Extract the letter from the original image with a 2-pixel margin around the edge
    letter_image = out[y - 2:y + h + 2, x - 2:x + w + 2]

    #resizing image for prediction
    letter_image = cv2.resize(letter_image, (30, 30))

    # Turn the single image into a 4d list of images
    letter_image = np.expand_dims(letter_image, axis=2)
    letter_image = np.expand_dims(letter_image, axis=0)

    # making prediction
    pred = model.predict(letter_image)

    # Convert the one-hot-encoded prediction back to a normal letter
    letter = lb.inverse_transform(pred)[0]
    predictions.append(letter)

    # draw the prediction on the output image
    cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 0, 255), 1)
    cv2.putText(output, letter, (x + 7, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

# get the captcha's text
captcha_text = "".join(predictions)

# Get the folder to save the image in
save_path = os.path.join(OUTPUT_FOLDER, captcha_text)

p = os.path.join(save_path + '.png')

#writing the image to the output folder
cv2.imwrite(p, image)
cv2.imwrite("{}/output.png" .format(OUTPUT_FOLDER), output)
# Show the annotated image
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

