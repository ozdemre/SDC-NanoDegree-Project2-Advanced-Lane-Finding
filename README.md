## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_chessboard.png "Chessboard Image Distortion"
[image2]: ./output_images/undistort_testimage.png "Raw Test Image Example"
[image3]: ./output_images/thresholding.png "Thresholding"
[image4]: ./output_images/perspective_transform.png "Perspective Transform"
[image5]: ./output_images/polynomial_fit.png "Polynomial Fit"
[image6]: ./output_images/radius_of_curvature.png "Reference image"
[image7]: ./output_images/draw_on_image.png "Final image"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook named "P4_ALF.ipynb" .  

First I imported all the necessary libraries for pipeline to work on both pictures and videos.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using 
the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

For this step I used camera calibration matrix that obtained in the beginning of the code. Simply I used:
```python
img = cv2.imread('test_images/test1.jpg') #TODO: try and save
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #imread reads in BGR

undistorted = cv2.undistort(img, mtx, dist, None, mtx)
```
to obtain undistorted test image. As camera lens is not causing too much distortion, it is hard to capture it at first look. But, it is there!
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Here is the code snippet, consist of magnitude, direction and HLS thresholds.
```python
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    else:
        print('error')
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    grad_binary = np.zeros_like(scaled_sobel)
    # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Apply threshold
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Apply threshold
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # Apply threshold
    return dir_binary

def combined_sobel_gradient_threshold(img,s_thresh_min=150,s_thresh_max=255):
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Apply each of the thresholding functions (gradient, magnitude and direction)
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))
    
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    # HLS Threshold
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # Threshold color channel

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(combined)
    combined_binary[(s_binary == 1) | (combined == 1)] = 1

    '''
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    f.tight_layout()
    ax1.imshow(img) # img is RGB
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Sobel Magnitude and Direction Threshold', fontsize=20)
    ax3.imshow(combined_binary, cmap='gray')
    ax3.set_title('hlS Threshold', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    '''
    return combined_binary, img
```

Here I basically applied the magnitude and direction threshold functions with the parameters that I used in the classroom quizzes. Then I also applied HLS thresholding for accounting different light condition.

After trying different configurations I have used gradient on x and y direction, manginutude of the gradient, direction of the gradient, and color transformation technique to get the final binary image.
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transform_image()`. Here I selected source and destination points
to get the birds eye view. FUnction returns wih transformed image and corresponding M and Minv transformation matrices for later use.

Here is the selected source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 470      | 200, 0        | 
| 717, 470      | 200, 680      |
| 260, 680     | 1000, 0      |
| 1043, 680      | 1000, 680        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points 
onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Since at this point we have a binary image where detected line pixels are represented as ones, we can compute 
histogram for bottom half of the image with a sliding window method starting from bottom of image and ending at top. 
This will give us identified line points which can be used for 2nd order polynomial fitting. This is done in `fit_polynomial()` function in my code.

Here are the parameters that I used for histogram computation with sliding window method.

```python
# HYPERPARAMETERS
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
```


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature calculation is implemented in `radius_curvature()` function.
Here radius of curvature of the curve at a particular point is defined as the radius of the approximating 
circle. This radius changes as we move along the curve. For calculation, I used the formula given in the classroom material.

![alt text][image6]


One additional calculation needs to be done is determining the position of the vehicle with respect to lanes. For that I used 3.7m lane width
```python
lane_center = (left_lane_bottom + right_lane_bottom)/2.
center_image = 640
center = (lane_center - center_image)*xm_per_pix #Convert to meters
position = "left" if center < 0 else "right"
center = "Vehicle is {:.2f}m {}".format(center, position)
```

`1095.10461584 891.236232627 Vehicle is 0.36m right`


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After lane identification, filled polylines and calculated values are written on warped image. then warped image is warped again for obtaining dash cam view. For this step I used same Minv transformation matrix..
This step is done in the `draw_on_image()` function given below.

```python
def draw_on_image(undist, warped_img, left_fit, right_fit, left_fitx, right_fitx, ploty, M, left_curvature, right_curvature, center, show_values = False):
    #ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0] )
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero)) #we should make it one channel
    #print(color_warp.shape)
    color_warp = np.zeros_like(warped_img).astype(np.uint8)
    #print(color_warp)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    Minv = np.linalg.inv(M)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    cv2.putText(result, 'Left curvature: {:.0f} m'.format(left_curvature), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Right curvature: {:.0f} m'.format(right_curvature), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, '{}'.format(center), (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    if show_values == True:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(result)
        
    return result
```
![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

After implementing all the pipeline steps, I created a `ProcessLine()` class for processing video feed. Same functions are applied in order and final video output is saved.
Algorith seems to work fairly good on video feed. Lines are correctly determined and radius of curvature is seems to be correct, at least in order of magnitude.  

Here's a [link to my video result](https://youtu.be/IvSUNDfjaII) 


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Compared to simple lane finding project (Project-1) this pipeline works better on curved roads under different lighting conditions. However, pipeline tuning (mainly thresholding) is done only considering this video feed.
It will probably not work well under different color lanes, at night, rain-snow or cases there are no lanes at all! And, it is very hard to generalize the tuning parameters for all possible conditions.
This project was a good example for traditional computer vision implementation. For deeper and more robust perception solutions deep learning and 
semantic segmentation could be a solution.
One additional thing I can (maybe should) make is a sanity check for processing frames. There are cases where, thresholding may fail or polynomial fitting may fail/poorly done which requires a sanity check in order to skip those frames and 
continue with the last good perception pipeline results. I initially created a function for this, but I did not have enough time to fully implement.

Some possible sanity checks (for further use maybe):
1. Lane parallelism
2. Radius of Curvature check both between lanes and thresholding with maximum-minimum values
3. Some sort of thresholding method for histogram calculation. (If enough number of lane pixels are not detected.)


Thanks Udacity team for this high level challenge.


