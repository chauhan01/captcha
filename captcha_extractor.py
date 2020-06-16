import argparse
import cv2
import numpy as np
import os
import glob
from keras.models import load_model
import pickle
from preprocessing import *

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image directory")
ap.add_argument("-save", "--save_solved", required=True,
	help="True if you want to save the solved output")
ap.add_argument("-m", "--model", required=True,
	help="path to model.hdf5 file")
ap.add_argument("-l", "--labels", required=True,
	help="path to captcha_labels.pickle file file")
args = vars(ap.parse_args())

test_image_folder = args['input']
OUTPUT_FOLDER_DETECTED = 'detected output'
OUTPUT_FOLDER_SOLVED = 'solved captcha output'

# if output directory does not exist creat one
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
LB = args['labels']
MODEL = args['model']
save_solved = args['save_solved']
test_images = glob.glob(os.path.join(test_image_folder, "*"))

#Loading the model
model = load_model(MODEL)

#load labels file
lb = open(LB, "rb")
lb = pickle.load(lb)
# loop over the image paths
for image in test_images:
    # Load the image and convert it to grayscale
    orignal_image = cv2.imread(image, cv2.IMREAD_COLOR)

    # Add some extra padding around the image
    image = cv2.copyMakeBorder(orignal_image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    img1, contours1 = image_process_1(image)
    img2, contours2 = image_process_2(image)
    img3, contours3 = image_process_3(image)
    
    if len(contours1) ==6:
        img = img1
        contours = contours1
    elif len(contours2) ==6:
        img = img2
        contours = contours2       
    else:
        img = img3
        contours = contours3
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

    # Creating an empty list for storing predicted letters
    predictions = []

    # Save each letter as a single image
    for letter_image_region, letter_bounding_box in zip(letter_image_regions, letter_bounding_rect):
        pts = np.array(letter_image_region, np.int32)

        mask = np.zeros((img.shape[0], img.shape[1]))

        cv2.fillConvexPoly(mask, pts, 1)
        mask = mask.astype(np.bool)

        out = np.zeros_like(img, np.uint8())
        out[mask] = img[mask]

        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = out[y - 2:y + h + 2, x - 2:x + w + 2]

        letter_image = cv2.resize(letter_image, (35, 35))

        # Turn the single image into a 4d list of images
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)
        letter_image = letter_image/255.0

        # making prediction
        pred = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(pred)[0]
        predictions.append(letter)

        # draw the prediction on the output image
        cv2.rectangle(image, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 0, 255), 1)
        cv2.putText(image, letter, (x + 7, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

    # Print the captcha's text
    captcha_text = "".join(predictions)

    save_path_detected = os.path.join(OUTPUT_FOLDER_DETECTED, captcha_text)
    detected_path = os.path.join(save_path_detected + '.png')
    cv2.imwrite(detected_path, image)

    if save_solved == True:
        save_path_solved = os.path.join(OUTPUT_FOLDER_SOLVED, captcha_text)
        solved_path = os.path.join(save_path_solved + '.png')
        cv2.imwrite(solved_path, orignal_image)
    else: Pass
