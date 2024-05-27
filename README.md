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