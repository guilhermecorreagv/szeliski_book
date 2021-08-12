# Chapter 3: Image Processing

## Question 3-1: Color balance

I made a simple application with the sliders to change the color balance, it simply multiplies by a factor between 0 and 1. 

<p align="center">
<img src="images/q3-1.png" width="30%">
</p>

To answer the questions:
1. It wouldn't change, since multiplying and taking an exponent of the factor, has the same value of multiplying by another number.
2. I can't perform this experiment.
3. If the foreground and background were taking with different color balances, we would have to twist one of them.

## Question 3-5: Difference keying

I analysed 10 frames to extract the mean and the standard deviation from the background, and also used some morphological post processing in order to reduce the noise in the image. I thresholded at 1.75 standard deviations from the background, and this was the result:

<p align="center">
<img src="images/q3-5.png" width="60%">
</p>
