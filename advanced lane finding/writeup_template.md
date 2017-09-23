# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[image1]: ./examples/undistorted_output.png "Undistort"
[image2]: ./examples/undistorted_test4.png "Road Transformed"
[image3]: ./examples/binaraized_test6.png "Binary Example"
[image4]: ./examples/warped_binarized_test6.png "Warp Example"
[image5]: ./examples/lane_fit.png "Fit Visual"
[image6]: ./examples/map_lane.png "Output"
[image7]: ./examples/processed_video.png "Video"
[video1]: ./processed_project_video.mp4 "Video"

---

### Writeup 

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `lines 34 through 60` in member method `calibrate_via_chessboards` of class `CameraCalibrator` in the file called `Advanced_Lane_Finding.py`).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `object_point` is just a replicated array of coordinates, and `object_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `image_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `object_points` and `image_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of L,S color channel (in HLS coloe space) and  Sobel operator in X direction  thresholds  to generate a binary image (thresholding steps at `lines 186`  in `Advanced_Lane_Finding.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()` in class `Perspective_Transformer`, which appears in lines 85 through 88 in the file `Advanced_Lane_Finding.py`. The constructor function of class `Perspective_Transformer` take source (`src_points`) and destination (`dest_points`) points. The `warp()` function takes as inputs an image (`image`). I chose the hardcode the source and destination points in the following manner in class member function `Line.build_perspective_transformer`:

```python
src = np.float32([[277, 670], [582, 457], [703, 457], [1046, 670]])
dst = np.float32([[277, 670], [277,0], [1046,0], [1046,670]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 277, 670      | 277, 670      | 
| 582, 457      | 277,0         |
| 703, 457      | 1046,0        |
| 1046, 670     | 1046,670      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
First I use histogram of pixels to find the bottm point of left and right lane lines of image, then I use 9 sliding windows to search the lane pixel from bottom to up. Recurrently  I use `+/-100 pixels margin` and `window_height` to make search window to search lane pixels and filter good lane pixel area by `min_num_pixels` threshhold. After extracted left and right line pixel positions, use np.polyfit to fit 2nd order polynomial curve kinda like this：

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
After fitting curve in pixels with form `f(y)= A*power(y,2)+B*y+C`, set `y= image_size[0]` , I get left_intersection and right_intersection which are the bottom pixel points of left and right lane in image. Then we use them to calculate length (in meters) per pixel and fit new polynomials to x,y in world space. Then I caculate the radius of curvature of the lane at the bottom Y position `np.max(ploty)`. The position of the vehicle with respect to center was caculated at pixel space and then map to world space by `xm_per_pix` scale factor.
I did this in `lines 350 through 386` in my code in `Advanced_Lane_Finding.py`



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `lines 388 through 411` in my code in `Advanced_Lane_Finding.py` in the function `fill_lane_lines()` and `merge_images()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./processed_project_video.mp4)

![alt text][image7]


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I use HLS color and X direction derivative  threshhold to binarizely filter pixel of lane line, this approach works when there is only yellow and white lane lines ,and the situation of road are perfect which means the rest of road excluded lane lines is all black.But in challenge video, the road are fixed which make extra unnormal lines on the road which makes lane line finding failed. 
On the other hand ,I use the histogram of pixel to find the bottom point in image of lane lines, which also makes failed detection, in future I will use instance Segmentation in deep learning algrthm to inprove de lane line area detection.


### Reference
My code refer to  code from this github https://github.com/upul/CarND-Advanced-Lane-Lines  which was helpful for me to complete my porject , thanks.
