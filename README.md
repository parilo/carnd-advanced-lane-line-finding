[![Lane line finding](http://img.youtube.com/vi/rZ_U9SQHMls/0.jpg)](https://www.youtube.com/watch?v=rZ_U9SQHMls)

This is project 4 of Udacity self-driving car nanodegree. The goal is to detect lane lines on video using computer vision methods (OpenCV library).

Code:
- All commented code can be found at Advanced-Lane-Line-Finding-v2.ipynb jupyter notebook.
- camera_cal/ folder contains camera calibration images
- test_images/ contains images of road to test and finetune pipeline
- project\_video.mp4, challenge\_video.mp4, harder\_challenge_video.mp4 - videos for pipeline testing

# Overview

Whole pipeline consist of following steps:

1. Removing camera distortion. We need camera calibration for that.
2. Filtering lane lines with morphology operations.
3. Applying Sobel operators to threshold lane lines pixels
4. Reprojecting front-view camera image to birds-eye view image
5. Fitting lane lines with parabola
6. Determining curvature of lane lines and car offset from center of the lane
7. Using information about previousely detected lane lines to simplify next lane lines detection
8. Drawing lane lines and reprojecting from birds-eye view back to front camera view along with printing curvature and offset information

# Camera calibration

Camera calibration is very importaint step. For successful camera calibration it's good to use bigger count of calibration images with wider variety of callibration pattern positioning. I did calibration with OpenCV library using cv2.calibrateCamera. Also OpenCV has very covenient methods for finding and drawing chessboard pattern. More about camera calibration may be found [here](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html). This is example of calibration images with visible distortion and corresponding undistorted images.

![camera calibration and undistortion] (https://github.com/parilo/carnd-advanced-lane-line-finding/blob/master/camera_calibration.png)

Road distorted (left) and undistorted (rigth) image.

![road image undistortion] (https://github.com/parilo/carnd-advanced-lane-line-finding/blob/master/undistorted_road_image.png)

# Morphology filter

To be able to threshold lane lines more correclty I decided to do prefiltering with "opening" morphology operation. Opening with horizontal kernel is used for separating all lane lines width or less width elements from the image. Also opening may be used for removing lane lines from picture if it is needed. More about morphology operations may be found [here](http://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html). As input for morphology filtering I use linear combination of grayscale and S channel from HLS color space representation.

```python
    gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    hls_s = get_s_from_hls (image)
    src = hls_s * 0.6 + gray * 0.4
```

![morphology filtering example] (https://github.com/parilo/carnd-advanced-lane-line-finding/blob/master/morphology-filter.png)

# Thresholding

After applying morphology filtering I find gradients on the image in different ways. I used horizontal gradient, vertical gradient, gradient magnitude, and two angle filtered gradients thresholds. All operations implemented using Sobel operators. To theshold lane lines pixels I use combination of mentioned gradients thresholds:

```python
    # horizontal part of gradient
    gradx = abs_sobel_thresh(src, orient='x', sobel_kernel=3, thresh=(1, 120))
    # vertical part of gradient
    grady = abs_sobel_thresh(src, orient='y', sobel_kernel=3, thresh=(1, 120))
    # magnitude of gradient
    mag_binary = mag_thresh(src, sobel_kernel=1, mag_thresh=(30, 100))
    # left line angle filtered gradient
    dir_binary_l = dir_thresh(src, sobel_kernel=ksize, thresh=(35*np.pi/180, 65*np.pi/180))
    # right line angle filtered gradient
    dir_binary_r = dir_thresh(src, sobel_kernel=ksize, thresh=(-65*np.pi/180, -35*np.pi/180))
    
    combined = np.zeros_like(src)
    # thresholding based on gradient combination
    combined[(
            ((gradx == 1) & (grady == 1)) | 
            (
                (mag_binary == 1) &
                ((dir_binary_l == 1) | (dir_binary_r == 1))
            )
        )] = 1
```

Here is an example of each gradient threshold and resulting thresholds combination.

![sobel thresholding] (https://github.com/parilo/carnd-advanced-lane-line-finding/blob/master/sobel_threshold.png)

# Birds-eye view

Considering road is near flat surface we can reproject it as plane from front camera view into birds-eye view using perspective projection. I use OpenCV cv2.warpPerspective for that. Needed projection matrix can be found with 4 points which is known to be on the road. Also it is possible to find inverse matrix that we can use to project back from birds-eye view into front camera view. Here is example of input and reprojected image along with mentioned 4 points.

```python
"""
Reproject from front view to top view
"""
def top_view (front_view_image):
    img_size = (front_view_image.shape [1], front_view_image.shape [0])
    return cv2.warpPerspective (
        front_view_image,
        # perspective transformation matrix
        M_perspective_720,
        img_size,
        flags=cv2.INTER_LINEAR
    )
```

![birds eye view perspective reprojection] (https://github.com/parilo/carnd-advanced-lane-line-finding/blob/master/birds_eye_view.png)

# Fitting lane lines

After reprojecting thresholded lane lines image into birds-eye view lane lines may be fitted with 2 degree polynomial or quadratic equation y = A*x**2 + B*x + C. I do it in three steps.

1. Calculating histogram of the image and extracting pixels around 2 highest maximums.
2. Applying [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) to fit quadratic equation into these pixels.
3. Filtering input pixels using RANSAC fitted curves.
4. Fitting quadratic equation with [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) method.

![birds eye view perspective reprojection] (https://github.com/parilo/carnd-advanced-lane-line-finding/blob/master/polinomial_fitting_steps.png)

# Curvature and car offset

After last step we got polinomial coefficients of lane lines. So we can calculate curvature and car offset from the canter of lane in meters. As we have undistorted our images we can measure distance between pixels. We need to know pixel horizontal size and vertical size in meters. It may be found at camera calibration step if you have information about calibration pattern size. For camera used in this project I already have information about pixel size in meters.

```python
 # meters per y pixel
ym_per_pix = 30./720

 # meters per x pixel 
xm_per_pix = 3.7/700

 # lane center position in meters
 # 1280 - image pixel width
center_position = 1280 * xm_per_pix / 2
```
Knowing these values we may recalculate polinomial coefficients, so curve will transfer from pixels to meters

```python
a = a * xm_per_pix / ym_per_pix**2
b = b * xm_per_pix / ym_per_pix
c = c * xm_per_pix
```
Lane line offset from left edge of the image and car offset from lane center can be calculated as
```python
y = 720 # image height, origin of ccord system in the top left corner
y = y * ym_per_pix
offset = (a*y*y + b*y + c)
 # car offset from lane center
car_offset = center_position - (offset_l + offset_r) * 0.5
```
since polynomial calculated as x = a*y**2 + b*y + c

Also curvature can be calculated as

```python
y = 720 # image height
y = y * ym_per_pix
try:
    curvature = pow(1 + (2*a*y + b)**2, 1.5) / math.fabs (2*a)
except ZeroDivisionError:
    curvature = 10000
```

For more information about radious of curvature see [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php)

# Example processed video frame image

Here is an example of resulted video frame image with drawn lane lines, curvature and offset information

![birds eye view perspective reprojection] (https://github.com/parilo/carnd-advanced-lane-line-finding/blob/master/drawn_lane_lines.png)

# Conclusion

Reprojecting into birds-eye view and fitting lane lines by polynomial is promising method which may find not only straight lane lines but also curved ones. But it is not robust enough to deal with complex environment with tree shadows, road defects, brightness/contrast issues. It will be effective on environments where lane lines are bright, contrast, not occluded or overlapped.

In many situations human drivers consider lane lines not as direct driving rule but as hint of which position of the road it is better to take. Also human drivers may predict lanes on roads without lane lines marks. I think this OpenCV approach may be used for generating dataset for more complex model training such as neural network which may be able to predict lane lines marks. Using well detected parts of video we may erase lane line marks using morphology operations and use such images without lane lines marks (and also another augmentation technics) along with detected curves as dataset for training. I think such method may be used to get lane lines on roads that even have no lane lines.

