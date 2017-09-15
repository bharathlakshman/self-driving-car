# **Finding Lane Lines on the Road**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[final]: ./examples/final.png "Final output"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline does the following tasks -
1. Convert the image/frame to grayscale
2. Blur the grayscale image with kernel of 11
3. Detect edges using Canny edge detection
4. Mask the image leaving out the region of interest(isosceles trapezium)
5. Find lines using Hough space transformation
6. Select the most dominant line on both left and right side of the lanes
7. Copy the most dominant lines to a cache to use it later to smoothen the lane movements ( I am using last know good location when lanes are not detected )
8. Draw an isosceles trapezium in Green color using the two most dominant lines(one left, one right)
9. Draw the most dominant lines in Bold Red
10. Iterate for each frame

![Detected Lane][final]


### 2. Identify potential shortcomings with your current pipeline


Tons of shortcomings -
1. We should consider speed of the vehicle for smoothening
2. We need to consider frame rate to average out the results
3. Last known good location needs to reset and blank out after serial lane detection failure
4. We need to extrapolate curvature as the lane curves around corners at different degrees (This works for only straight lanes/free ways with banked roads)
5. Region of interest has to dynamically increase or decrease it's dimensions based on elevation of the road
6. Lane detection will fail when a big truck passes by causing massive shadows
7. It will fail when one or more neighboring vehicles are driven on the white markings
8. It will fail when vehicles in front the our vehicle also has straight lines or markings resembling the ones on road
9. It will fail to pick up lanes when sun directly shines on the road eliminated all possible distinctions
10. Poor lane clarity with rain can also be an issue.
I can see several other ways it can fail.

### 3. Suggest possible improvements to your pipeline

1. Region selection needs to be dynamic
2. Canny edge arguments needs to change based on lighting
3. Lines needs to be extrapolated to curves
4. Speed and frame rate needs to be considered for averaging
5. Last known lane needs to fail after few failed frames
Several performance improvements can be done to speed up the processing.
