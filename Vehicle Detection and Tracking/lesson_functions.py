import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        file_features = single_img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)
        features.append(np.concatenate(file_features))

        # Augment dataset with flipped image
        feature_image=cv2.flip(feature_image,1) 
        file_features = single_img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)
    # Return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def find_cars(img, ystart, ystop, xstart, xstop, scale, step):
    boxes = []
    draw_img = np.zeros_like(img)   
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))       
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) -1
    cells_per_step = step  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_features = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            # Extract the image patch
            subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))        
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)+xstart
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((int(xbox_left), int(ytop_draw+ystart)),(int(xbox_left+win_draw),int(ytop_draw+win_draw+ystart))))
    return boxes

'''
import numpy as np
import cv2
from skimage.feature import hog
'''
def convert_color(img, conv='RGB2LUV'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

#######################From project Advanced lane finding###############################

import glob
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

# store camera calibration parameters in ./camera_cal/calibrated_data.p
CALIBRATED_DATA_FILE = './camera_cal/calibrated_data.p'

class CameraCalibrator:
    def __init__(self, glob_regex, x_corners, y_corners, init_coef=True):
 
        
        # images used for camera calibration
        self.calibration_images = glob_regex
        
        # The number of horizontal corners in calibration images
        self.x_corners = x_corners
        
        # The number of vertical corners in calibration images
        self.y_corners = y_corners
        self.object_points = []
        self.image_points = []

        self.calibrated_data = {}
        if not init_coef:
            self.calibrate_via_chessboards()

        self.coef_loaded = False

    def calibrate_via_chessboards(self):

        object_point = np.zeros((self.x_corners * self.y_corners, 3), np.float32)
        object_point[:,:2] = np.mgrid[0:self.x_corners, 0:self.y_corners].T.reshape(-1, 2)

        for idx, file_name in enumerate(self.calibration_images):

            image = mpimg.imread(file_name)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_image,(self.x_corners, self.y_corners),None)
            if ret:
                self.object_points.append(object_point)
                self.image_points.append(corners)

				# Draw and display the corners
                #cv2.drawChessboardCorners(gray_image, (9,6), corners, ret)
                #self.chessboards.append(gray_image)

        h, w = image.shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.image_points, (w, h), None, None)

        self.calibrated_data = {'mtx': mtx, 'dist': dist}

        with open(CALIBRATED_DATA_FILE, 'wb') as f:
            pickle.dump(self.calibrated_data, file=f)
		
        self.coef_loaded = True

    def undistort(self, image):

        if not os.path.exists(CALIBRATED_DATA_FILE):
            raise Exception('Camera calibration data file does not exist at ' + CALIBRATED_DATA_FILE)

        if not self.coef_loaded:

            with open(CALIBRATED_DATA_FILE, 'rb') as fname:
                self.calibrated_data = pickle.load(file=fname)

            self.coef_loaded = True
	
        return cv2.undistort(image, self.calibrated_data['mtx'], self.calibrated_data['dist'],None, self.calibrated_data['mtx'])

class Perspective_Transformer:

    def __init__(self, src_points, dest_points):

        self.src_points = src_points
        self.dest_points = dest_points

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dest_points)
        self.M_inverse = cv2.getPerspectiveTransform(self.dest_points, self.src_points)

    def warp(self, image):

        size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.M, size, flags=cv2.INTER_LINEAR)

    def unwarp(self, src_image):

        size = (src_image.shape[1], src_image.shape[0])
        return cv2.warpPerspective(src_image, self.M_inverse, size, flags=cv2.INTER_LINEAR)


def noise_reduction(image, threshold=4):

    # This method is used to reduce the noise of binary images.

    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(image, ddepth=-1, kernel=k)
    image[nb_neighbours < threshold] = 0
    return image

def binary_threshold_filter(channel, thresh = (200, 255), on = 1):
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = on
    return binary

###########################################################################################
# This method extracts lane line from warped image by creating a binarized image 
# where lane lines in white color and rest of the image in black color.
###########################################################################################

def binary_pipeline(image, 
					hls_s_thresh = (170,255),
					hls_l_thresh = (30,255),
					hls_h_thresh = (15,100),
					sobel_thresh=(20,255),
					mag_thresh=(70,100),
					dir_thresh=(0.8,0.9),
					r_thresh=(150,255),
					u_thresh=(140,180),
					sobel_kernel=3):



    # Make a copy of the source iamge
    image_copy = np.copy(image)

    gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

    r_thresh = (150,255)
	# RGB colour
    R = image_copy[:,:,0]
    G = image_copy[:,:,1]
    B = image_copy[:,:,2]
    rbinary = binary_threshold_filter(R, r_thresh)
    
    u_thresh = (140,180)
    # YUV colour
    yuv = cv2.cvtColor(image_copy, cv2.COLOR_RGB2YUV)
    Y = yuv[:,:,0]
    U = yuv[:,:,1]
    V = yuv[:,:,2]
    ubinary = binary_threshold_filter(U, u_thresh)

    # Convert RGB image to HLS color space.
    # HLS more reliable when it comes to find out lane lines
    hls = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]
    h_channel = hls[:, :, 0]
    # We generated a binary image using S component of our HLS color scheme and provided S,L,H threshold
    s_binary = binary_threshold_filter(s_channel,hls_s_thresh)
    l_binary = binary_threshold_filter(l_channel,hls_l_thresh)
    h_binary = binary_threshold_filter(h_channel,hls_h_thresh)

    # We apply Sobel operator in X,Y direction and calculate scaled derivatives.
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = binary_threshold_filter(scaled_sobel,sobel_thresh)

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobely = np.absolute(sobely)
    scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    sybinary = binary_threshold_filter(scaled_sobel,sobel_thresh)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    mag_binary = binary_threshold_filter(gradmag, mag_thresh)

    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = binary_threshold_filter(absgraddir, dir_thresh)

    # Return the combined binary image
    binary = np.zeros_like(sxbinary)
    binary[((l_binary == 1) & (s_binary == 1) | (sxbinary == 1))] = 1
    binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')

    return noise_reduction(binary)

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):

        self.detected = False

        self.left_fit = None
        self.right_fit = None

        self.MAX_BUFFER_SIZE = 12
        self.IMAGE_HIGH = 720
        self.buffer_index = 0
        self.iter_counter = 0

        self.buffer_left = np.zeros((self.MAX_BUFFER_SIZE, self.IMAGE_HIGH))
        self.buffer_right = np.zeros((self.MAX_BUFFER_SIZE, self.IMAGE_HIGH))

        self.perspective = self.build_perspective_transformer()
        self.calibrator = self.build_camera_calibrator()

    def build_perspective_transformer(self):

        src = np.float32([[277, 670], [582, 457], [703, 457], [1046, 670]])
        dst = np.float32([[277, 670], [277,0], [1046,0], [1046,670]])

        perspective = Perspective_Transformer(src, dst)
        return perspective

    def build_camera_calibrator(self):

        glob_regex = glob.glob('./camera_cal/calibration*.jpg')
        calibrator = CameraCalibrator(glob_regex,9, 6, init_coef=False)
        return calibrator

    def init_lane_finder(self, binary_warped):

		# Bottom half region of image  of 0 channel
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :, 0], axis=0)

        # get midpoint of the histogram i.e half of width
        midpoint = np.int(histogram.shape[0] / 2)

        # get left and right half points of the histogram
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9

		# Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

		# Set the width of the windows +/- margin
        margin = 100
		# Set minimum number of pixels found to recenter window
        min_num_pixels = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            
			# Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
			# Search pixels in sliding windows
            good_left_inds = (  (nonzeroy >= win_y_low) 
								& (nonzeroy < win_y_high) 
								& (nonzerox >= win_xleft_low) 
								& (nonzerox < win_xleft_high)).nonzero()[0]

            good_right_inds = (	(nonzeroy >= win_y_low) 
								& (nonzeroy < win_y_high) 
								& (nonzerox >= win_xright_low) 
								& (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > min_num_pixels:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > min_num_pixels:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

		# Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx  = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
        
		# First detect lane lines
        self.detected = True

        return left_fitx, right_fitx

    def recurrent_lane_finder(self, binary_warped):

        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 100

        left_lane_inds = ( (nonzerox > ( self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] - margin)) 
							& (nonzerox < ( self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = ( (nonzerox > ( self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] - margin)) 
							& (nonzerox < ( self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        return left_fitx, right_fitx

    def calculate_road_parameters(self, image_size, left_x, right_x):

        # This method calculates left and right lain line curvature and distance of the vehicle off from the center

		# f(y)= A*power(y,2)+B*y+C
		# y= image_size[0]  ==> f(y) = intersection points at the bottom of our image
        left_intersection = self.left_fit[0] * image_size[0] ** 2 + self.left_fit[1] * image_size[0] + self.left_fit[2]
        right_intersection = self.right_fit[0] * image_size[0] ** 2 + self.right_fit[1] * image_size[0] + self.right_fit[2]

        # Caculate the distance in pixels between left and right intersection points
        road_width_in_pixels = right_intersection - left_intersection

        # Here means something wrong with lane finding 
        assert road_width_in_pixels > 0, 'Road width in pixel must be positive!!'

        # Since average highway lane line width in US is about 3.7m
        # Assume lane is about 30 meters long and 3.7 meters wide
        # we calculate length per pixel and ensure "order of magnitude" correct
        xm_per_pix = 3.7 / road_width_in_pixels
        ym_per_pix = 30 / image_size[0]

        # Calculate road curvature in X-Y space
        ploty = np.linspace(0, image_size[0]-1, num=image_size[0])
        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_x * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_x * xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        # Calculate the lane deviation of the car
        calculated_center = (left_intersection + right_intersection) / 2.0
        lane_deviation = (calculated_center - image_size[1] / 2.0) * xm_per_pix
        return left_curverad, right_curverad, lane_deviation

    def fill_lane_lines(self,image, fit_left_x, fit_right_x):

        #This method highlights correct lane section on the road
        copy_image = np.zeros_like(image)

        fit_y = np.linspace(0, copy_image.shape[0] - 1, copy_image.shape[0])

        pts_left = np.array([np.transpose(np.vstack([fit_left_x, fit_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_right_x, fit_y])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(copy_image, np.int_([pts]), (0, 255, 0))

        return copy_image

    def merge_images(self, binary_img, undistorted_src_img):

        copy_binary = np.copy(binary_img)
        undistorted_copy_src_img = np.copy(undistorted_src_img)
        result = cv2.addWeighted(undistorted_copy_src_img, 1, self.perspective.unwarp(copy_binary), 0.3, 0)
        return result

    ####################################################################
    # Advance lane line finding pipeline
    #1. Make undistorted image
	#2. Warp undistorted image
	#3. Binarize image through color and gradient threshhold
	#4. Use sliding window to find left and right lanes
	#5. Use buffer history found lanes to smooth the current lanes
    #6. Caculate the left and right lane curvatures and vehicle offset of the center of the lane
    #7. Unwarp processed image back and display
    ####################################################################

    def image_process_pipeline(self, image):

        image = np.copy(image)
        undistorted_image = self.calibrator.undistort(image)
        warped_image = self.perspective.warp(undistorted_image)
        binary_image = binary_pipeline(warped_image)

        if self.detected:
            left_fitx, right_fitx = self.recurrent_lane_finder(binary_image)
        else:
            left_fitx, right_fitx = self.init_lane_finder(binary_image)

        self.buffer_left[self.buffer_index] = left_fitx
        self.buffer_right[self.buffer_index] = right_fitx

        self.buffer_index += 1
        self.buffer_index %= self.MAX_BUFFER_SIZE

        if self.iter_counter < self.MAX_BUFFER_SIZE:
            self.iter_counter += 1
            ave_left = np.sum(self.buffer_left, axis=0) / self.iter_counter
            ave_right = np.sum(self.buffer_right, axis=0) / self.iter_counter
        else:
            ave_left = np.median(self.buffer_left, axis=0)
            ave_right = np.median(self.buffer_right, axis=0)

        left_curvature, right_curvature, lane_deviation = self.calculate_road_parameters(binary_image.shape, ave_left, ave_right)
        left_lane_curv_text = 'Estimated left Curvature  : {:.2f} m'.format(left_curvature)
        right_lane_curv_text = 'Estimated right Curvature : {:.2f} m'.format(right_curvature)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(undistorted_image, left_lane_curv_text, (50, 50), font, 1, (250, 250, 0), 2)
        cv2.putText(undistorted_image, right_lane_curv_text, (50, 90), font, 1, (250, 250, 0), 2)

        deviation_text = 'Estimated lane Deviation: {:.3f} m'.format(lane_deviation)
        cv2.putText(undistorted_image, deviation_text, (50, 130), font, 1, (250, 250, 0), 2)

        filled_image = self.fill_lane_lines(binary_image, ave_left, ave_right)
        merged_image = self.merge_images(filled_image, undistorted_image)

        return merged_image

    def get_undistorted_image_with_lane_lines(self, image):

        img = np.copy(image)
        image_with_lane = self.image_process_pipeline(img)

        return image_with_lane