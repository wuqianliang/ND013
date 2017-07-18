#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip


# Global variables used to reduce jitter between frames
SET_LFLAG=0
SET_RFLAG=0
LAST_LSLOPE=0
LAST_RSLOPE=0
LAST_LEFT = [0, 0, 0]
LAST_RIGHT = [0, 0, 0]

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def reset_global_vars():

	global SET_LFLAG
	global SET_RFLAG
	global LAST_LSLOPE
	global LAST_RSLOPE
	global LAST_LEFT
	global LAST_RIGHT


	#reset start
	SET_RFLAG=0
	SET_LFLAG=0
	LAST_LSLOPE=0
	LAST_RSLOPE=0
	LAST_LEFT = [0, 0, 0]
	LAST_RIGHT = [0, 0, 0]
	#reset end

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):

	global SET_RFLAG
	global SET_LFLAG
	global LAST_LSLOPE
	global LAST_RSLOPE
	global LAST_LEFT
	global LAST_RIGHT


	right_y_set = []
	right_x_set = []
	right_slope_set = []

	left_y_set = []
	left_x_set = []
	left_slope_set = []
	slope_min=0.35
	slope_max=0.85
	middle_x = img.shape[1] / 2
	max_y = img.shape[0]
	for line in lines:
		for x1,y1,x2,y2 in line:
			fit = np.polyfit((x1, x2), (y1, y2), 1)
			slope=fit[0]
			if slope_min < np.absolute(slope) <= slope_max:
				#left_top (0,0)  right_bottom(max_x,max_y)
				#right lane lines
				if slope > 0 and x1 > middle_x and x2 > middle_x:
					right_y_set.append(y1)
					right_y_set.append(y2)
					right_x_set.append(x1)
					right_x_set.append(x2)
					right_slope_set.append(slope)
				#left lane lines
				elif slope < 0 and x1 < middle_x and x2 < middle_x:
					left_y_set.append(y1)
					left_y_set.append(y2)
					left_x_set.append(x1)
					left_x_set.append(x2)
					left_slope_set.append(slope)

	COR_DRIFT_WEIGHT=0.7
	SLOPE_DRIFT_WEIGHT=0.9
	#draw left line
	if left_y_set:
		lindex = left_y_set.index(min(left_y_set)) #top point
		left_x_top = left_x_set[lindex]
		left_y_top = left_y_set[lindex]
		lslope = np.median(left_slope_set)
		if SET_LFLAG >0:
			lslope = lslope + (LAST_LSLOPE - lslope)*SLOPE_DRIFT_WEIGHT

		left_x_bottom = int(left_x_top + (max_y - left_y_top) / lslope )

		if SET_LFLAG >0:
			left_x_bottom = int(LAST_LEFT[0] + (left_x_bottom - LAST_LEFT[0])*COR_DRIFT_WEIGHT)
			left_x_top = int(LAST_LEFT[1] + (left_x_top - LAST_LEFT[1]) * COR_DRIFT_WEIGHT)
			left_y_top = int(LAST_LEFT[2] + (left_y_top - LAST_LEFT[2]) * COR_DRIFT_WEIGHT)
		else:
			SET_LFLAG=1

		cv2.line(img, (left_x_bottom, max_y), (left_x_top, left_y_top), color, thickness)
		LAST_LSLOPE=lslope
		LAST_LEFT=[left_x_bottom,left_x_top,left_y_top]

	#draw right line
	if right_y_set:
		rindex = right_y_set.index(min(right_y_set)) #top point
		right_x_top = right_x_set[rindex]
		right_y_top = right_y_set[rindex]
		rslope = np.median(right_slope_set)

		if SET_RFLAG >0:
			rslope=rslope + (LAST_RSLOPE - rslope)*SLOPE_DRIFT_WEIGHT

		right_x_bottom = int(right_x_top + (max_y - right_y_top) / rslope )
		

		if SET_RFLAG >0:
			left_x_bottom = int(LAST_RIGHT[0] + (left_x_bottom - LAST_RIGHT[0]) * COR_DRIFT_WEIGHT)
			left_x_top = int(LAST_RIGHT[1] + (left_x_top - LAST_RIGHT[1]) * COR_DRIFT_WEIGHT)
			left_y_top = int(LAST_RIGHT[2] + (left_y_top - LAST_RIGHT[2]) * COR_DRIFT_WEIGHT)
		else:
			SET_RFLAG=1

		cv2.line(img, (right_x_top, right_y_top) , (right_x_bottom, max_y) , color, thickness)
		LAST_RSLOPE=rslope
		LAST_RIGHT=[left_x_bottom,left_x_top,left_y_top]
		

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def weighted_img(img, initial_img, alpha=0.8, beta=1., labda=0.):
	return cv2.addWeighted(initial_img, alpha, img, beta, labda)

def process_image(image):
	"""
	1. reading in an image
	2. Define a kernel size and apply Gaussian smoothing
	3. Canny Edge Detection
	4. create a masked edges image 
	5. detect lines via Hough Transformations
	6. draw the lane lines onto the original image
    """

	# Get grayscale
	gray = grayscale(image)

	# Define a kernel size and apply Gaussian smoothing
	kernel_size = 5
	blur_gray = gaussian_blur(gray,kernel_size)

	# Define our parameters for Canny and apply
	canny_low_threshold = 75
	canny_high_threshold = canny_low_threshold*3
	edges = canny(blur_gray, canny_low_threshold, canny_high_threshold)

	# Create a masked edges image 
	#vertices = np.array([[(0,540),(450, 316), (512,319), (905,529)]], dtype=np.int32)
	imshape = image.shape
	vertices = np.array([[(0,imshape[0]), (9*imshape[1]/20, 11*imshape[0]/18), (11*imshape[1]/20, 11*imshape[0]/18), (imshape[1],imshape[0])]], dtype=np.int32)
	masked_edges = region_of_interest(edges,vertices)


	# Define the Hough transform parameters
	rho = 1					# distance resolution in pixels of the Hough grid
	theta = np.pi/180		# angular resolution in radians of the Hough grid
	hof_threshold = 20		# minimum number of votes (intersections in Hough grid cell)
	min_line_len = 30		# minimum number of pixels making up a line
	max_line_gap = 60	    # maximum gap in pixels between connectable line segments
	lines=hough_lines(masked_edges, rho, theta, hof_threshold, min_line_len, max_line_gap)

	# Make a blank the same size as our image to draw on
	line_image = np.copy(image)*0
	# Iterate over the output "lines" and draw lines on a blank image
	draw_lines(line_image, lines,thickness=10)


	# Create a "color" binary image to combine with line image
	color_edges = np.dstack((edges, edges, edges)) 

	# Draw the lines on the edge image
	alpha=0.8
	beta=1.
	lambda_=0.
	lines_edges=weighted_img(image, line_image, alpha, beta, lambda_)
	return lines_edges
	

# save processed images to directory test_images_output
imageNames = os.listdir('test_images/')
for image_name in imageNames:
	reset_global_vars()
	image = mpimg.imread("test_images/{}".format(image_name))
	plt.imsave("test_images_output/output_{}".format(image_name), process_image(image))

reset_global_vars()
videoNames = os.listdir('test_videos/')
for video_name in videoNames:
	reset_global_vars()
	clip1 = VideoFileClip("test_videos/{}".format(video_name))
	white_clip = clip1.fl_image(process_image)
	white_clip.write_videofile("output_{}".format(video_name), audio=False)
