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
    binary[(((l_binary == 1) & (s_binary == 1) | (sxbinary == 1)) ) ] = 1
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

        corners = np.float32([[277, 670], [582, 457], [703, 457], [1046, 670]])

        src = np.float32([corners[0], corners[1], corners[2], corners[3]])
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

    def merge_images(self, binary_img, src_image):

        copy_binary = np.copy(binary_img)
        copy_src_img = np.copy(src_image)

        copy_binary_pers = self.perspective.unwarp(copy_binary)
        result = cv2.addWeighted(copy_src_img, 1, copy_binary_pers, 0.3, 0)

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

        left_curvature, right_curvature, lane_deviation = self.calculate_road_parameters(image.shape, ave_left, ave_right)
        left_lane_curv_text = 'Estimated left Curvature  : {:.2f} m'.format(left_curvature)
        right_lane_curv_text = 'Estimated right Curvature : {:.2f} m'.format(right_curvature)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(image, left_lane_curv_text, (50, 50), font, 1, (250, 250, 0), 2)
        cv2.putText(image, right_lane_curv_text, (50, 90), font, 1, (250, 250, 0), 2)

        deviation_text = 'Estimated lane Deviation: {:.3f} m'.format(lane_deviation)
        cv2.putText(image, deviation_text, (50, 130), font, 1, (250, 250, 0), 2)

        filled_image = self.fill_lane_lines(binary_image, ave_left, ave_right)
        merged_image = self.merge_images(filled_image, image)

        return merged_image

if __name__ == '__main__':


    line = Line()
    output_file = './processed_project_video.mp4'
    input_file = './project_video.mp4'
    clip = VideoFileClip(input_file)
    out_clip = clip.fl_image(line.image_process_pipeline)
    out_clip.write_videofile(output_file, audio=False)
