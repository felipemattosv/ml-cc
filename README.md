# ml-cc
Keep exercises and studies during [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course?hl=pt-br).

## ML Terminology
- Label (`y`): variable to predict;
- Features (`X`): input variables;
- Example: particular instance of data. Can be labeled (X, y) or unlabeled (X, ?). Labeled examples are used to train the model;
- Model: maps examples to predicted labels y'. A regression model predicts continuous values and a classification model predicts discrete values.

## Linear Regression
Determine a equation for a line that shows the relationship between N features and labels using the labeled examples.

$ y' = b + w_1 \cdot x_1 + ... + w_N \cdot x_N  $

where:
- `b` (or $ w_0 $): bias (y-intercept)
- `w`: weight related to each feature

## Training and Loss
Training a model means learning(determining) good values for weights and the bias from labeled examples. In supervised learning, a ml algorithm builds a model by examining many examples and attempting to find a model that minimize loss (penalty for a bad prediction). This process is called empirical risk minimization.

### Squared Loss ($ L_2 $ loss)
Common used in linear regression models.

The $ L_2 $ loss for a single example is: $ (y - prediction(x))^2 = (y - y')^2 $

### Mean square error (MSE)
Is the average squared loss per example over all N dataset examples.

$ MSE = 1/N \cdot \sum_{i=0}^{N} (y_i - y'_i)^2 $

## Reducing Loss

### Gradient Descent
The gradient of a function is a vector that points in the direction of the greatest rate of increase of the function, and whose magnitude is that rate of increase. So if we want to minimize the loss function, we have to go in the opposite direction of the gradient.

To calculate the gradient, we have to calculate the derivative of the loss function with respect to the weights and biases.

So we reapeatedly take small steps in the direction that minimizes the loss.

But compute the gradient over the entire dataset on each step is unnecessary and will spend a lot of computation.So we have techniques to make this more efficiently:

#### Stochastic Gradient Descent (SGD)
Look at only one example for step.

#### Mini-Batch Stochastic Gradient Descent
Look at random batches (groups of 10-1000 examples) for step, then average the Loss & Gradients over the batch.

### Weights Initialization
Usually the loss function isn't convex, i.e, has more than one minimun. This means that the training performance has strong dependency on initial values.

### Hyperparameters
Hyperparameters are the configuration settings used to tune how the model is trained.

Examples are the BATCH_SIZE, LEARNING_RATE, ...

### Learning rate
Gradient descent algorithm multiply the gradient by a scalar known as learning rate (also sometimes called step size) to determine the next point.

To perform a good training we have to used the right learning rate. If the LR is too small, learning will take too long. Conversely, if the LR is too large, the next point will perpetually bounce across the bottom of the loss function.

## Generalization
Generalization refers to your model's ability to adapt properly to new, previously unseen data, drawn from the same distribution as the one used to create the model.

### Overfitting
When the model learns details and noise from training data so well that it loses the ability to generalize to new data. This means that the model may perform excellently on training data, but perform poorly on previously unseen data.

A good way to stop overfitting is reducing the models complexity and removing noise and outliers from the training dataset.

### Diving the dataset
A nice way to see if the model is good for generalization is getting a sample of the data and using it only for testing. Good performance on the test set is a useful indicator of good performance on the new data in general, assuming that:
- the test-set is large enough
- you don´t cheat by using the same test set over and over

This separation must be random, stationary (i.e, it doesn´t change over time) and always from the same data distribution.

## Data partitioning scheme
To avoid overfitting in test data, comes a new partitioning called validation-set.

- TRAINING-SET: Used to train the model.

- VALIDATION-SET: Used to evaluate model during training and adjust hyperparameters.

- TEST-SET: Ignored during training, it is only used to evaluate the model generalization performance (i.e, check if the model is overfitting to the validation-set)

