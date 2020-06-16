import numpy as np
import cv2



# creating a function to remove blobs
def remove_blobs(image):
    
    #find all connected components (white blobs in the image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    min_size = 65 

    #creating clean image
    img = np.zeros((output.shape))
    #for every component in the image, we keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img[output == i + 1] = 255

    img = np.uint8(img)
    return(img)


def image_process_1(image):
    
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # applying blur
    median = cv2.medianBlur(gray,1)

    # applying threshold
    thresh = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]
 
    # applying erode
    kernel = np.ones((2,2),np.uint8)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
        
    img = remove_blobs(thresh)
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    return img, contours
    


def image_process_2(image):
    
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # applying blur
    median = cv2.medianBlur(gray,1)

    # applying threshold
    thresh = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]
    img = remove_blobs(thresh)
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    return img, contours



def image_process_3(image):
    
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # applying blur
    median = cv2.medianBlur(gray,1)

    # applying threshold
    thresh = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]
    
    # applying erode
    kernel = np.ones((2,2),np.uint8)


    img = cv2.erode(thresh,kernel,iterations = 1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  
    
    # img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)       
    img = remove_blobs(img)
    img= cv2.medianBlur(img,3)
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return img, contours

