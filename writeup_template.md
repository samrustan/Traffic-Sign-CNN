# **Traffic Sign Recognition** 

## Write-up

#### Samuel Rustan
#### February Cohort

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points

In the following headings, the [rubric points](https://review.udacity.com/#!/rubrics/481/view) will be addressed at each point in my implementation.  

---
### Github link
Here is a link to my [project code](https://github.com/samrustan/Traffic-Sign-CNN

### Data Set Summary & Exploration

The code for this step is contained in the 2nd and 3rd code cell of the IPython notebook.  

Python libraries: pandas, numpy, and matplotlib were used to plot a sample set of the training image set.

A Histogram of the data was also plotted using a bar graph to show the distribution of the image set.  A histogram for the training, validation, and test set are displayed.

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text](https://github.com/samrustan/Traffic-Sign-CNN/distribution.png?raw=true)

### Design and Test a Model Architecture

The code for this step is contained in the fifth code cell of the IPython notebook.

I ran a normalization on the set such that computationally this is more stable and would decrease processing time.

 norm = (img_array - img_array.min()) / (img_array.max() - img_array.min())
 
In addition to normalization, I shuffled the data before running each EPOCH to distribute the probability that the model doesn't train on a specific pattern in the data.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? 

The training set contained 34799 examples.

The validation set contained 4410 examples.  This is about 11% of the total data available for training.  The training and validation sets were provided as such and no splitting was necessary.

The test set contained 12630 examples.

Under Model Architecture is where I set the training BATCH_SIZE, EPOCHS, and LEARNING_RATE to the same values as in the Labs. 

Due to time constraints I did not augment the data, though it is possible that augmenting the data by adding artificial shifts and rotations to teach the network how to recover from poor position or orientation does add undesirable artifacts as the magnitude increases. --Excerpted from Nvidia end-to-end paper.  This is something that I'm going to explore further.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 

The model archictecture is the LeNet model straight from the Lab.  Max Pooling has been applied at the Convnet layers.

To avoid overfitting, Dropout was applied on the feed-forward layers.  Dropout randomly zeroes out values and creates a probability that the activation will be passed forward.  By applying dropout, the network must confirm the values it produces repeatly to avoid generating noise.  Effectively, the network cannot rely on the "regularity" of the dataset.

Also, L2 regularization and Cross Entropy on weights was applied to penalize large weight ("weight-gain") momentum shifts.  I used an L2_strength value of 1e-4.  This was the largest value suggested by the Amazon AWS training parameters suggestion.

http://docs.aws.amazon.com/machine-learning/latest/dg/training-parameters.html

My final model consisted of the following layers:

| Layer           |     Description                               | 
|:---------------:|:---------------------------------------------:| 
| Input           | 32x32x3 RGB image                             | 
| Convolution 3x3 | 1x1 stride, VALID padding, outputs 32x32x64   |
| ReLu            |                                               |
| Max pooling     | 2x2 stride, VALID padding, outputs 32x32x64   |
| Convolution 3x3 | 1x1 stride, VALID padding, outputs 32x32x64   |
| ReLu            |                                               |
| Max pooling     | 2x2 stride, VALID padding, outputs 32x32x64   |
| Flattened       |                                               |
| Fully Connected | shape=(1600, 1024), mean = 0, stddev = 0.1    |
| ReLu            |                                               |
| Dropout         | keep probability: 50%                         |
| Fully Connected | shape=(1024,512), mean = 0, stddev = 0.1      |
| ReLu            |                                               |
| Dropout         | keep probability: 50%                         |
| Fully Connected | shape=(512, 43), mean = 0, stddev = 0.1       |
| ReLu            |                                               |
| Dropout         | keep probability: 50%                         |
| Softmax         |                                               |

2 convnet, Relu, max pool layers
1 flatten
3 Fully Connected, Relu, with Dropout.
 
#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* EPOCHS = 20
* BATCH_SIZE = 128
* LEARNING_RATE = 0.001
* Adam Optimizer

The code for training the model is located in the tenth cell of the ipython notebook. 

To train the model, I used a training-validation-test solution.  I chose a relatively low number of epochs for a couple reasons, I wanted to know the least amount of training that would be required to train the model.

The training examples are batched in sizes of 128 and fed to the model in a for loop.  Training accuracy is compared to validation accuracy and after 20 Epochs, the training accuracy reached 100% by the 20th Epoch and the validation acurracy was at 97%.  This gives me some pause as it seems too high for the low amount of epochs I used.  

I plotted the result of the training and validation accuracy, as well as the training and validation loss.  The plot show the training accuracy approaching a steady state very quickly, though the trend is smooth without any jumps.  From the lectures this seems to be a good thing and gives some confidence in the model.  The validation plot is not as smooth, though trends in kind with the training set.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* Training Accuracy of 99.99%
* Validation set accuracy of 96.96%
* Test set accuracy of 95.48%

If a well known architecture was chosen:

* What architecture was chosen? 

The LeNet architecture from the lab was implemented.  

* Why did you believe it would be relevant to the traffic sign application?

Seemed a likely candidate from the suggestion by David Silver in the video...

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Well, they're high.  Though it's clear that doesn't mean much since in the next section where I tested on new images it seemed to fail pretty well.  
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![test1][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

1 of the 5 images were classified correctly, though it seemed to fail on a speed limit sign, which I would think should be an easy sign to classify.  Only one sign was classified correctly.  The top_k_preds.index shows the confidence for the one it got right to be 100%.  There was a "General Caution" image that was in the predicted set, however, it was the wrong image.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


Here are the results of the prediction:

| Image			              |     Prediction	     | Confidence | 
|:---------------------:|:-------------------:|:----------:| 
| Children crossing    	| General caution     |	  69.3%	   | 
| General caution 		    | Priority road       |	  25.3%	   |
| No entry				          | Priority road		     |	  43.4%	   |
| Roundabout mandatory  | Roundabout mandatory|	  100%	    |
| Speed limit (60km/h)  | Slippery Road       |   98.3%	   |


The results are somewhat confusing, since there was only a single correct prediction that gave 100%, yet there was another predicton at 98% that was totally wrong.  The Speed limit sign seems like an fairly easy sign to classify too.

The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This doesn't compare at all to the test set accuracy of 95%.  Clearly, there's something that's either not correctly implemented or the test results are not indicative of the models success, indicating that there might be problems with the model itself.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is fairlt sure that this is a "General Caution" --which is incorrect, and it's only 3% sure that it is a Children's Crossing --which it IS!!  I'm very much confused by this result.

| Probability  |     Prediction	        					| 
|:------------:|:---------------------------:| 
| 69%         	| General Caution   							   |  
| 10%     				 | Roundabout Mandatory 							|
|  5%					     | No Passing          								|
|  3%	      			| Children Crossing		  		 				|
|  2%				      | Dangerous Curve to the right|


### Conclusions:

Overall this was a very challenging project for me as my programming skill and understanding of CNNs were fully taxed.  It's clear to me that my model is not working correctly and I'm not clear on where I should focus my efforts to improve the model.  I've tried several iterations of trainingm using upto 100 epochs and have found no consistent improvements in performance.  Since this is only the second project, I'm hoping to gain further insight on the next projects ahead.
