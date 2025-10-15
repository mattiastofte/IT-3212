## PCA
We selected the "sea" folder of images from the list of grayscales. There were some issues with the size of these images and the limit of our computers processing power when calculating the
eigenvalues and eigenvectors, but more on that later. The result of this is that in the normalization step we also rescaled the images to be smaller. We tested  a lot of sizes, and 80x80 was the largest we could go before the program wouldn't complete the calculations and crash.
### Step 0:
The dataset also contained some color pictures and to avoid sorting through each individually we loaded all pictures as grayscale, then resized the pictures. For the Normalizing we flattened the image and used MinMaxScaler from sklearn to ensure the pixel values were between 0 and 1. Finally we reshaped the image and added it to a list of normalized images. We also had to reduce the sample size for the same processing issues as previously mentioned, for the final run we used 50 images out of the 2000+.
### Step 1:
To convert the images into a 2d matrix we loaded the images into an np array and used .reshape to flatten each image.
### Step 2:
For the next step we used np.cov on the flattened images to create a covariance matrix. However, this resulted in the images never fully reconstructing in the end. To solve this we added a step for centering the data. by subtracting the mean image. To underline the importance of this, here is the reconstruction with various k values.
![[Pasted image 20251015145529.png]]
Above is the reconstruction with k set to 1000 and we can see that regardless of how high the k value is, the picture is still a mess. 
Below is the reconstruction with k=100.
![[Pasted image 20251015145752.png]]
Though the sample is not the same, there is an obvious difference in the image quality. This can also be seen in the MSE values.
### Step 3:
Calculating the eigenvalues and -vectors is pretty straight forward with np.linal.eigh()
### Step 4:
First we generate a list of sorted indices based on the eigenvalues and flip it to get the indices in descending order. Then we sort both the eigen values and vectors based of these indices. 
### Step 5 & 6:
For this step we chose k = 40 to maintain some variance while still getting a final result of a mostly reconstructed image. We select the components then run project the images by running a dot product operation with the original centered data on the components.
## Reconstruction
To reconstruct the images back into "original space" we need to again perform a dot operation, now with the projected data and a transposed version of the same components. Additionally we add back the mean image we subtracted earlier. 
To properly observe the difference over multiple dimensions we performed the same actions as above with a variety of k values and displayed them in order as can be seen in the pictures earlier.

## Experimentation
For good measure, here is another set of images:
![[Pasted image 20251015154909.png]]
We see that, as mentioned earlier, there is clear improvement in quality based on k value with little to no improvement past 50. This is also reflected in the CEV graph. ![[Pasted image 20251015155932.png]]
Where we see a clear stagnation around the k=50 area. Lowering the k value max further we get a little better quality: ![[Pasted image 20251015160652.png]]
## Visual analysis
See the images earlier in the report. 
As mentioned before we see a clear loss of visual quality when k is below 40. Using the provided images we can see a clear loss of detail when comparing k20 to k40. Using the image of a dock in the first image set column 4, we see that at k20 the boats might as well be rocks even if the general shapes are maintained.
## Error Analysis:
Here are a list of calculated MSE values:![[Pasted image 20251015161518.png]]
We see, as previously mentioned, a clear progression where the image quality plateaus around 40. Where the quality of the image is good enough that details are clearly discernable while not pushing the MSE to 0. We can conclude that any k value with an MSE score of less than 0.001 provides acceptable image quality.