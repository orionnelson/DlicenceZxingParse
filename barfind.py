# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from rembg.bg import remove
import io
from PIL import Image
debug = False
import warnings
warnings.filterwarnings("ignore")





#Pre Image Processing Before Find Barcode is Used Taken From https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# Future Use

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")



def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped




def cropreturn(image):
    temp2 = 'temp2.png'
    ratio = image.shape[0] / 300.0
    orig = image.copy()
    image = imutils.resize(image, height = 300)
    cv2.imwrite(temp2,image)
    imgc = np.fromfile(temp2)
    result = remove(imgc)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    img.save(temp2)
    image = cv2.imread(temp2)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 75, 400,4)
    if debug: cv2.imshow("edged",edged)
    if debug: cv2.waitKey(0)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #print(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    biggest = 0
    bigcontour = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.0225 * peri, True)
        rect = cv2.minAreaRect(approx)
        #rect = (rect[0],(rect[1][0]+12,rect[1][1]),rect[2]) # (center(x, y), (width, height), angle of rotation) 
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(c)
        #print(area) 
        if (area > biggest and len(approx)==4):
                biggest = area
                bigcontour = box
                screenCnt= (approx)
                lbox = order_points(box)
                #print("Set New Card")
        if(bigcontour is not None):
               cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
               if debug: cv2.imshow("Card Found", image)
               if debug: cv2.waitKey(0)
               #print(rect)
    #print(screenCnt)
    if screenCnt is not None:
            upright = True
            #print(lbox[0])# Top Left 
            #print(lbox[1])# Top Right
            #print(lbox[2])# Bottom Right
            #print(lbox[3])# Bottom Left
            #print(lbox[0][0])
            top = cv2.norm(lbox[0],lbox[1],cv2.NORM_L2)
            #print(top)
            rs = cv2.norm(lbox[1],lbox[2],cv2.NORM_L2)
            sideways = top<rs
            #print(rs)
            temp = lbox[0]
            if(sideways):
                    dst_pts = np.array([[0, 0],   [1080, 0],  [1080, 1920], [0, 1920]], dtype=np.float32)
            else:
                    dst_pts = np.array([[0, 0],   [1920, 0],  [1920, 1080], [0, 1080]], dtype=np.float32)
                    
            #print(top)
            #print(rs)
            #print("Lbox above")
            screenCnt = (lbox * ratio)
            screenCnt = np.float32(screenCnt)        
            M = cv2.getPerspectiveTransform(screenCnt, dst_pts)
            # Card is Sideways
            if(sideways):
                    warp = cv2.warpPerspective(orig, M, (1080, 1920))
                    if(lbox[0][1] > lbox[1][1]):
                            warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
                    else:
                            warp = cv2.rotate(warp, cv2.ROTATE_90)
            else:
                    warp = cv2.warpPerspective(orig, M, (1920, 1080))
            if debug: cv2.imshow("Card Found", warp)
            if debug: cv2.waitKey(0)
            if(warp is not None):
                    return warp
            else:
                Exception("There was somthing wrong with the function fml")
                return
    else:
         return orig
    
    

    






def findbarcode(image):
    image = cropreturn(image)
    def automatic_brightness_and_contrast(image, clip_hist_percent=25):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        '''
        # Calculate new histogram with desired range and show histogram 
        new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
        plt.plot(hist)
        plt.plot(new_hist)
        plt.xlim([0,256])
        plt.show()
        '''

        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return (auto_result, alpha, beta)
    

    # construct the argument parse and parse the arguments
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-i", "--image", required = True,
    #        help = "path to the image file")
    #args = vars(ap.parse_args())
    # load the image and convert it to grayscale
    #image = cv2.imread(args["image"])
    dim = (int(1920),int(1080))

    try:
        #print(image.shape)
            
        if((image.shape[0] < 1080) and (image.shape[1]<1920)):
            image = cv2.resize(image,dim,cv2.INTER_NEAREST)
            #print('cube')
        else:
            image = cv2.resize(image,dim,cv2.INTER_AREA)
            #print('big')
        #print(image.shape)
        early = image.copy()
        auto_result, alpha, beta = automatic_brightness_and_contrast(image)
        image = auto_result
        
    except Exception as e:
        #print(image.shape)
        print(e)
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #template = cv2.imread('biggesttemp.jpg',0)
    template = cv2.imread('Template\\template.png',0)
    w, h = template.shape[::-1]
    graycopy = gray.copy()
    res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.0425
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        closed2 = cv2.rectangle(gray, pt, (pt[0] + w, pt[1] + h), (0,0,0), -1)
    if debug: cv2.imshow("Patterm Match", closed2)
    (_, closed3) = cv2.threshold(closed2, 1, 255, cv2.THRESH_BINARY)
    if debug: cv2.imshow("Patterm Match2", gray)
    #invert closed 3
    closed4 = cv2.bitwise_not(closed3)
    if debug: cv2.imshow("Patterm Match3", closed4)
    gray = cv2.bitwise_and(graycopy,closed4,closed4)
    if debug: cv2.imshow("rejectedmask", gray)
    
    cv2.waitKey(0)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 2.4
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # blur and threshold the image
    blurred = cv2.blur(gradient, (8, 8))
    (_, thresh) = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 8)
    closed = cv2.dilate(closed, None, iterations = 8)
    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #issue is lack of approximation
    i=0
    maxarea=0
    maxcontour = None
    while i < len(contours):
        #print("here")
        c = contours[i]    
        area = cv2.contourArea(c)
        #print(area)
        #print("maxarea")
        #print(maxarea)
        if( area < 1000):
                contours.pop(i)
        else:
                if area > maxarea:
                        maxcontour = c
                        #print(str(maxarea) + "area" + str(i))
                        maxrect = cv2.minAreaRect(c)
                        maxarea = area
                        #print(maxrect)
                else:
                        pass
                i = i + 1
    spotted = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
    if debug: cv2.imshow("found contours", spotted)
    #cnts = imutils.grab_contours(cnts)
    #c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    # compute the rotated bounding box of the largest contour
    if debug: cv2.imshow("Image", closed)
    if debug: cv2.waitKey(0)
    #print(image.shape)
    #print(maxrect)
    rect = maxrect
    x, y, w, h = cv2.boundingRect(maxcontour)
    ah, aw = 25, 135 # padding for bounding box because rotation comes next
    roi = early[y-int(ah/2):int(y+h+ah/2), x-int(aw/2):int(x+w+aw/2)]
    if debug: cv2.imshow('ROI',roi)
    # Perform transformation using the angle of the barcode note
    #sometimes cv is stupid and the barcode is given a rotation of ~90 if the rectangle uses its largest edge as height. 
    angle = maxrect[2] # 2 specifies angle
    if angle > 45 :
            angle -=90
    
    rows,cols = roi.shape[0], roi.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(roi,M,(cols,rows), borderValue=(255,255,255))

    if debug: cv2.imshow('Rotated image',img_rot)
    cimg = img_rot




    
    #print(rect)
    #print("rect above")
    #box = cv2.boxPoints(rect)
    #cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    
    #print(rect)
    box3 = cv2.boxPoints(rect)
    box3 = np.int0(box3)
    
    spotted2 = cv2.drawContours(image.copy(), [box3], -1, (0, 255, 0), 3)
    if debug: cv2.imshow("spot 2", spotted2 )
    obx = order_points(box3)
    if debug: cv2.imshow("early", early)
    
    auto_result, alpha, beta = automatic_brightness_and_contrast(cimg)
    cimg = auto_result
    cv2.waitKey(0)
    cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    height, width = cimg.shape[:2]
    #Closing using Morph and OSU
    cimg2 = cv2.morphologyEx(cimg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    #print(cimg2.shape)
    dens = np.sum(cimg2, axis=0)
    mean = np.mean(dens)
    #print(mean)
    #Shady stuff
    thresh = cimg2.copy()
    for idx, val in enumerate(dens):
        if val< 10800:
            thresh[:,idx] = 0
    (_, thresh2) = cv2.threshold(thresh, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if debug: cv2.imshow("OTSU Mean Adjusted Image", thresh2)
    output = thresh2
    if debug: cv2.waitKey(0)
    return output
