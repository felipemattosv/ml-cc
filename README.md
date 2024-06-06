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

<img src="assets/gradient_descent.png" alt="Gradient descent" width="400"/>

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

| Overfitted model after training | Overfitted model during test |
|----------|----------|
| <img src="assets/overfitted_model_after_training.png" alt="Overfitted model after training phase" width="300"/> | <img src="assets/overfitted_model_during_test.png" alt="Overfitted model during test" width="300"/>) |

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

OBS: Before splitting the data, is good to suffle the dataset to prevent the dataset from being ordered by some feature that influences the label.

## Feature engineering
Mapping Raw Data to Features.

### Mapping numeric values
Trivial because the feature will have the same value of the raw data

Example:

(raw data) num_rooms: 6 => (feature) num_rooms_feature  = 6.0

### Mapping categorical string values
Since models cannot multiply strings by the learned weights, we use feature engineering to convert strings to numeric values.

Example:

(raw data) street_name: ['Charleston Road', 'Shorebird Way', 'Rengstorff Avenue', others...]

- 'Charleston Road' => 0
- 'Shorebird Way' => 1
- 'Rengstorff Avenue' => 2
- others (Out-Of-Vocabulary bucket) => 3

(feature) street_name_feature: [0, 1, 2, 3...]

However, if we incorporate these index numbers directly into our model, it will impose some constraints that might be problematic.

- If the model learn one weight to street_name_feature, it will be multiplied for different values. Our model needs the flexibility of learning different weights for each street.

- If the house is in a corner, we can´t represent this in the structure above.

Solution:

#### one-hot encoding

- For values that apply to the example, set correspong vector element to `1`
- Set all other elements to `0`

The lenght of this vector is equal to the number of elements in the vocabulary.

This approach effectively creates a Boolean variable for every feature possible value (e.g., street name)

So if the value is 0, it will turn the value * weight product to 0. And if the example has two values, then both will be `1` and the rest will be zero (e.g., if the house is in the corner, the two binary values related to the streets are set to 1, and the model will uses both their respective weights).

(raw data) street_name: 'Shorebird Way' => (feature) street_name_feature = [0, 1, 0, 0...]

##### Sparse representation
Suppose that you had 1,000,000 different street names in your data set that you wanted to include as values for street_name. Explicitly creating a binary vector of 1,000,000 elements where only 1 or 2 elements are true is a very inefficient representation in terms of both storage and computation time when processing these vectors. In this situation, a common approach is to use a sparse representation in which only nonzero values are stored. In sparse representations, an independent model weight is still learned for each feature value, as described above.

## Qualities of Good Features

### Avoid rarely used discrete feature values

Example: "house_type" is a good feature because it´s possible values (['victorian', 'modern', 'kitnet']) will reapeat over the examples and the model will be abble to recognize it´s patterns. But a feature like "unique_house_id" is a bad feature because each value would be used only once, so the model couldn't learn anything from it.

### Prefer clear and obvious meanings

Example: "house_age_years: 27" (years) is better for debbug than "house_age: 851472000" (time since unix)

### Don´t mix "magic" values with actual data

Example: imagine that we have a 0 to 1 float feature like "quality_rating". It´s NOT a good ideia to represent with -1 when the user didn´t entered the quality rating. Instead, it´s a better way creating a feature named "is_quality_rating_defined" and set it to 0.

In the original feature, replace the magic value as follows:
- For variables that take a finite set of values (discrete variables), add a new value to the set and use it to signify that the feature value is missing.
- For continuous variables, ensure missing values do not affect the model by using the mean value of the feature's data.

### Don´t choose values that can possibly change in the future to represent a feature

## Cleaning data
Good ML relies on good data.

### Scalling feature values
Scaling means converting floating-point feature values from their natural range (for example, 100 to 900) into a standard range (for example, 0 to 1 or -1 to +1). Feature scalling provides the following benefits:

- Helps gradient descent converge more quickly
- Helps avoid the "NaN trap" (values exceeds the floating-point precision limit during training)
- Helps the model learn appropriate weights for each feature. Without feature scaling, the model will pay too much attention to the features having a wider range.

#### Linerly map [min value, max value] to a small scale, such as [-1, 1]

#### Z score

$ scaledvalue = (value - mean) / stddev $

### Handling extreme outliers
First of all we need to find the outliers, for that, is useful to plot the Probability Density Function of the feature and see if it has a "tail". A long "tail" means that are some huge outliers. 

<img src="assets/outliers.png" alt="Outliers example" width="300"/>

To minimize the effect of this outliers, we can apply the $ log $ on the feature or set the feature to $ feature = min(feature, L) $ where $ L $ is the value that we will clip the outliers.

### Binning
If we´re working with latitude, for example, is useful to separate latitude in bins,

example: LatitudeBin1 = 32 < latitude <= 33; (...); LatitudeBin6 = 37 < latitude <= 38;

<img src="assets/binning.png" alt="Binning example" width="500"/>

Doing that, instead of having a floatting-point feature, we now have 11 distinc boolean features, that we can unite to a single 11-element-vector. Doing that we can avoid rarely used discrete feature values and our model can learn weights for each region.

### Scrubbing
Clean data that are not trustworthy, for example,
- ommited values
- duplicate examples
- wrong labels
- wrong feature values (sensor readding errors, ...)

## Feature Crosses
A feature cross is a synthetic feature formed by multiplying (crossing) two or more features. With feature crosses we can input non-linearity to linear models.

## Regularization
Regularization means penalizing the complexity of a model to reduce overfitting.

Before the goal of the training optimization algorithm was only reducing loss:

$ minimize(Loss(Data, Model)) $

Now, we want to continue minimizing the loss, but minimize also the complexity of the model:

$ minimize(Loss(Data, Model) + complexity(Model)) $

We can quantify `complexity` using the **$ L_2 $ regularization**, which defines the regularization term as the sum of the squares of all the feature weights:

$ L_2 $ $ regularization = w_1^{2} + w_2^{2} + ... + w_n^{2} $

In this formula, weights close to zero have little effect on model complexity, while outlier weights can have a huge impact.

This approach follows the Bayesian prior:
- weights should be centered around zero
- weights shoud be normally distributed

### Lambda (Regularization rate)

$ minimize = (Loss(Data, Model) + λ * complexity(Model)) $

$ = (Loss(Data, Model) + λ * (w_1^{2} + ... + w_n^{2})) $

Note that the second term doesn´t depend on the data and the two terms are balanced with the `λ` coefficient, which says how much we care about learning w/ the training data Versus making a simple model.

## Logistic Regression
Instead of predicting exactly 0 or 1, logistic regression generates a probability—a value between 0 and 1, exclusive.

This approach is useful for calculating the probability and to map into a binary classification problem.

### Sigmoid function
Restrict(squeezes) the output to always falls between 0 and 1.

$ \large σ(z) = \frac{1}{1 + e^{-z}} $, where `z` is the output of a the linear layer ($ z = b + W * X $)

<img src="assets/sigmoid.png" alt="Sigmoid function graph" width="450"/>

This function basically turns every negative number into a value near to zero, every positive number into a value near to one and increases 'constantly' around the input zero.

Note that `z` is also referred to as the log-odds because the inverse of the sigmoid states that `z` can be defined as the log of the probability of the `1` label divided by the probability of the `0` label.

$ z = log(\frac{σ}{1 - σ}) $

### Log Loss: Loss function for Logistic Regression
the loss function for linear regression is squared loss. The loss function for logistic regression is the **Log Loss**, which is defined as follows:

$ \large Log Loss = \sum_{(x,y) ∈ D} -y * log(y') - (1 - y) * log(1 - y') $

where:

-  `(x,y) ∈ D`: data set containing many labeled examples ($ (x,y) $ pairs).
- `y`: label in the labeled example (`0` or `1`).
- ``y'`: predicted value (between `0` and `1`)

### Regularization in Logistic Regression
Regularization is extremely important in logistic regression modeling. Without regularization, the asymptotic nature of logistic regression would keep driving loss towards 0 in high dimensions.
