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

![test1][data:image/jpeg;base64,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] ![alt text][image5] ![alt text][image6] 
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
