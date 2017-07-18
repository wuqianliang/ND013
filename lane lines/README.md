# **Finding Lane Lines on the Road** 


### Reflection

### 1. Describe of pipeline.

My pipeline consisted of 5 steps. 
1. Reading in an image
2. Define a kernel size and apply Gaussian smoothing
3. Canny Edge Detection
4. Create a masked edges image 
5. Detect lines via Hough Transformations
6. Draw the lane lines onto the original image


In order to draw a single line on the left and right lanes, I use Canny Edge Detection and Hough Transforms to detect lines in an image, and according to the slope and X coordinate of the line segments, the line segment is divided into left and right lines. Then I use average slope to reduce jitter between frames.

<img src="test_images_output/output_solidWhiteRight.jpg" width="480" alt="Weighted Image" />


### 2. shortcomings

In the optional challenge video, the lanes are some jittering where there are some shadows. 

The actual lane is curved, and lines I painted is straight.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use some fitering algorithm to avoid the effects in lanes detecting.

Another potential improvement could be to use Spline curve algorithm to draw curved lane lines in challenge video.
