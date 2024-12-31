# Introduction To Time Series: A comprehensive guide to time series in machine learning
![Banner](assets/banner_time_series.png)

# Table of Contents

- [Fundamental Step: Time Indexing](#fundamental-step-time-indexing)
  - [MultiIndexing in Pandas](#multiindexing-in-pandas)
    - [Before Time Indexing](#before-time-indexing)
    - [After Time Indexing](#after-time-indexing)
- [Model Architectures (1)](#model-architectures-1)
  - [Linear Regression](#linear-regression)
    - [Regularization](#regularization)
      - [Ridge (L2)](#ridge-l2)
      - [Lasso (L1)](#lasso-l1)
- [Adding Temporal Dependencies using Deterministic Processes](#adding-temporal-dependencies-using-deterministic-processes)
  - [Linear Regression](#linear-regression-1)
  - [Decision Trees](#decision-trees)
  - [Deterministic Processes](#deterministic-processes)
    - [Introducing Seasonality](#introducing-seasonality)
      - [One Hot Encoding](#one-hot-encoding)
      - [Fourier Calendar Terms](#fourier-calendar-terms)
- [Algorithm Complexity Note](#algorithm-complexity-note)
  - [Linear Regression](#linear-regression-2)
  - [XGBoost](#xgboost)

# Fundamental Step: Time Indexing

Always time index your data! Without time indexing the model has no idea how different features relate. You essentially just pass 1 large input vector  (time seres) into your data. Ensure that before you even start looking at model architectures, your data is time indexed.

## MultiIndexing in Pandas

You can use pandas to time index your data with the method of multiindexing. In proper termss, this takes your input data from a "long" format to a "wide one". This step is crucial in any part of time series. Below is an exmaple. 

### Before Time Indexing

Here we have an example of before time indexing our data. In this even though it looks like we have "features" Store ID and product family, this isnt the case! These "features" get completely ignored in the model! All we pass in as data is one long time series of sales. Essentially just one vector for our model to learn from.

| Date       | Store ID | Product Family | Sales |
|------------|----------|----------------|-------|
| 2017-01-01 | 1        | Frozen Pizza   | 50.0  |
| 2017-01-01 | 1        | Bicycles       | 5.0   |
| 2017-01-01 | 2        | Frozen Pizza   | 30.0  |
| 2017-01-01 | 2        | Bicycles       | 8.0   |
| 2017-01-02 | 1        | Frozen Pizza   | 52.0  |
| 2017-01-02 | 1        | Bicycles       | 6.0   |
| 2017-01-02 | 2        | Frozen Pizza   | 29.0  |
| 2017-01-02 | 2        | Bicycles       | 7.0   |

### After Time Indexing

This is the wide format. We time index our data, and now pass in 1782 vectors ($54 \text{ product families} * 33 \text{ store numbers}$)!

| Date       | Frozen Pizza (Store 1) | Bicycles (Store 1) | Frozen Pizza (Store 2) | Bicycles (Store 2) |
|------------|-------------------------|--------------------|-------------------------|--------------------|
| 2017-01-01 | 50.0                   | 5.0                | 30.0                   | 8.0                |
| 2017-01-02 | 52.0                   | 6.0                | 29.0                   | 7.0                |

# Model Architectures (1)

## Linear Regression

Linear regression models the relationship between dependent variable ($y$) and one or more independent variables ($x_1, x_2, x_3, ... , x_n$). The goal of linear regression is to find a line that best fits the data by minimizing the error between predicted values and the actual values. The equation is presented below:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \epsilon
$$

Where:

- $y$ is the dependent variable (target)
- $x$ is the independent variable (predictor)
- $\beta_0$ Intercept of the line
- $\beta_n$ Slope for each corresponding predictor $x_n$
- $\epsilon$ Error term

We make certain assumptions about the data when implementing a linear regression model:

- Linearity: The relationship between y and x is linear. (Note: we can still model non linear relationships between x and y! When we say that we assume linearity, we model the way x and y combine to be linear, not necessarily that the relationship between x and y is linear!)
- Independence: The residuals ($Y_i - \hat{Y}_i$) show be independent of each other. In other words, the error of one observation should not depend on the error of another observation. In time series, errors may often show autocorrelation, where residuals depend on previous time points! We can check this by using the Durbin-Watson test, a statistical terst used to detect autocorrelation in residuals.
- Homoscedasticity: The variance of residuals should be constant across all levels of the independent variables. We can detect heteroedasticity (unequal variance) by plotting residuals vs predicted values. We may also use statistical tests like the Breusch-Pagan test or White's test. 
- Normality - Residuals hsoul dbe approximately normally idstriubted. This ensures that coefficient estimates are unbiased and confidence intervals and p-values are reliable. We can check this by plotting a histogram or performing the Shapiro-Wilk test or Kolmogorov-Smirnov test.
- No Multicollinearity: Independent variables ($X_1, X_2, ... , X_n$) should not be highly correlated with each other. When 2 or more independent variables are highly correlated, they carry redundant information, which makes it hard to determine their individual effect on the target variable $Y$. We can detect multicollinearity by calculating a ocrrelation matrix or calculating the variance inflation factor.

The most commonly used loss function for Linear Regression is the Mean Squared Error (MSE), which is defined as:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2
$$

where:
- $Y_i$ is the actual value,
- $\hat{Y}_i$ is the predicted value.

To minimize the MSE, we solve for the optimal coefficients $\beta$ using Ordinary Least Squares (OLS). The predicted values are given by:

$$
\hat{Y} = X\beta 
$$

where:
- $X$ is the design matrix (input features),
- $\beta$ is the vector of coefficients,
- $\hat{Y}$ is the vector of predictions.

The goal is to minimize the Residual Sum of Squares (RSS):

$$
RSS = (Y - X\beta)^T (Y - X\beta) = Y^TY - Y^T(X\beta) - (X\beta)^TY + (X\beta)^T(X\beta)
$$

which simplifies to:

$$
RSS = Y^TY - 2(X\beta)^TY + B^TB(X^TX)
$$

Taking the derivative of RSS with respect to $\beta$ and setting it to zero gives:

$$
\frac{Y^TY}{\partial \beta} = 0
$$
$$
\frac{2(X\beta)^TY}{\partial \beta} = 2X^TY
$$
$$
\frac{B^TB(X^TX)}{\partial \beta} = 2(X^TX)\beta
$$

$$
\frac{\partial RSS}{\partial \beta} = -2X^T (Y - X\beta) = 0 
$$

Solving for $\beta$, we obtain the normal equation:

$$ 
\beta = (X^T X)^{-1} X^T Y 
$$

This equation provides the optimal coefficients $\beta$, which minimize the squared error. This relationship is very important. In MSE, we will never have to iteratively optimize for a solution!

### Regularization

#### Ridge (L2)

We can avoid overfitting and multicollinearity in our linear regression model by using Ridge (L2). How it works is by adding penalties at the end of our loss function:

$$
Loss = MSE + Penalty = \frac{1}{n} \sum_{i=1}^{n} {(Y_i - \hat{Y}_i)^2} + \lambda \sum \beta^2
$$

##### Intermezzo : Linear Algebra 

Lets take a step back to dive deeper into how the linear algebra here. From our equation of OLS:

$$ 
\beta = (X^T X)^{-1} X^T Y 
$$

Where $X$ is the design matrix, containing all the predictor variables:

$$
\begin{bmatrix}
X_{11} & X_{12} & \dots & X_{1p} \\
X_{21} & X_{22} & \dots & X_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
X_{n1} & X_{n2} & \dots & X_{np}
\end{bmatrix}
$$

where:
- each row corresponds to a single data point ($i = 1,2,...,n$),
- each column corresponds to a feature or predictor ($j = 1,2,...,n$)
- $X_{ij}$ represents the value of the j-th predictor for the i-th observation.

Let's consider an example to make this more concrete. Consider a house price prediction problem:
- Row i: represents a specific house,
- Column j: represents a specific feature of the house, size, number of rooms or age.

$$
\begin{bmatrix}
Size & No. of Rooms & Age \\
1200 & 3 & 10 \\
2000 & 5 & 20
\end{bmatrix}
$$

This is our input matrix. We then have to calculate $X^TX$:

$$
\begin{bmatrix}
(X^TX)\_{11} & (X^TX)\_{12} & \dots & (X^TX)\_{1p} \\
(X^TX)\_{21} & (X^TX)\_{22} & \dots & (X^TX)\_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
(X^TX)\_{n1} & (X^TX)\_{n2} & \dots & (X^TX)\_{np}
\end{bmatrix}
$$

To calculate $(X^TX)$ if the input design matrix $X$ is an $nxp$ matrix, with n observations and p features, our output gram matrix will have shape $pxp$. To calculate the value of our gram matrix at some position $jk$, we follow the formula and do some linear algebra operations:

$$
X = 
\begin{bmatrix}
X_{11} & X_{12} & \dots & X_{1p} \\
X_{21} & X_{22} & \dots & X_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
X_{n1} & X_{n2} & \dots & X_{np}
\end{bmatrix}
$$

$$
X^T = 
\begin{bmatrix}
X_{11} & X_{21} & \dots & X_{n1} \\
X_{12} & X_{22} & \dots & X_{n2} \\
\vdots & \vdots & \ddots & \vdots \\
X_{1p} & X_{2p} & \dots & X_{np}
\end{bmatrix}
$$

For two matrices A and B, the element at position ($j, k$) in their multiplication is:

$$
(AB)_{jk} = \sum\_{\text{over common index}} \text{Row}_j (A) \cdot \text{Col}_k (B)
$$

To compute $(X^TX)_{jk}$, we take the dot product of j-th row of $X^T$ and the k-th column of $X$. Our result:

$$
(X^TX)_{jk} = \sum\_{i=1}^n X\_{ij}X\_{ik}
$$

Lets consider an example to make this more clear visually. Consider design matrix $X$:

$$
X = 
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6 \\
\end{bmatrix}
$$

$$
X^T = 
\begin{bmatrix}
1 & 3 & 5 \\
2 & 4 & 6
\end{bmatrix}
$$

$$
X^TX= 
\begin{bmatrix}
(1)(1) + (3)(3) + (5)(5) & (1)(2) + (3)(4) + (5)(6) \\
(2)(1) + (4)(3) + (6)(5) & (2)(2) + (2)(4) + (6)(6)
\end{bmatrix} =  
\begin{bmatrix}
35 & 44 \\
44 & 56
\end{bmatrix}
$$

The variance measures the spread or variability of a single feature:

$$
Var(X_j) = \frac{1}{n} \sum_{i=1}^n (X_{ij} - \bar{X}_j)^2
$$

where:
- $X_{ij}$ value of the j-th feature for observation i
- $\bar{X}_j$ mean of the j-th feature

The covariance measures the relationship between two features $X_j$ and $X_k$:

$$
Cov(X_j, X_k) = \frac{1}{n} \sum_{i=1}^n (X_{ij} - \bar{X}_j)(X\_{ik}-\bar{X_k})
$$

We assume that the features are mean centered, from our equation for $(X^TX)_{jk}$, we find that for diagonal elements:

$$
(X^TX)_{jj} = \sum\_{i=1}^n X\_{ij}^2
$$

Which is proportional to our variance formula if we were mean centered! For off-diagonal elements $(j \neq k)$:

$$
(X^TX)_{jk} = \sum\_{i=1}^n X\_{ij}X\_{ik}
$$

Which is also proportional to our covariance formula! We have that:

$$
Var(X_j) = \frac{1}{n}(X^TX)_{jj}
$$

$$
Cov(X_j, X_k) = \frac{1}{n}(X^TX)_{jk}
$$

Note also the properties of our gram matrix, and the determinant being very close to 0. If we have matrix:

$$
\begin{bmatrix}
Var(X_1) & Cov(X_1,X_2) \\
Cov(X_2,X_1) & Var(X_2)
\end{bmatrix}
$$

We get the determinant is close to 0 with the following relationship:

$$
Cov(X_1,X_2) \approx \sqrt{Var(X_1)*Var(X_2)}
$$
$$
Det(X^TX) = Var(X_1)*Var(X_2)-Cov(X_1,X_2)Cov(X_2,X_1) 
$$

We can say that a matrix is ill-conditioned, or nearly singular, when the ratio of largest to smallest eigen value is greater than $1\cdot10^6$:

$$
\kappa(X^TX) = \frac{\sigma_{max}}{\sigma_{min}}
$$

Kappa is known as the condition number.

##### Connecting Back

After all this linear algebra, what does this have to do with L2 regularization? How does this help multicollinearity and overfitting?

Multicollinearity occours when two or more predictors are highly correlated. Consider the case where one predictor is perfectly correlated to another predictor ($X_1 = 2X_2$), in this case we get an eigenvalue of 0, which means that our gram matrix is singular. If predictors are highly correlated, then we get eigenvalues very close to 0, resulting in high conditioning numbers, which results in a model that is very sensitive to noise in the train set.

In ridge regression we add a penalty term lambda, in this case our equation for OLS results to:

$$ 
\beta = (X^T X + \lambda I)^{-1} X^T Y 
$$

This results in our eigenvalues being increased by lambda:

$$
\sigma_{j}^\text{ridge} = \sigma_j + \lambda
$$

This stablizes our inversion, helping with predictors being highly correlated. As we also penalise large $\beta$ coefficients, we avoid specific predictors being highly used, helping with overfitting!

#### Lasso (L1)

With lasso regularization, we have the advantes of automatic feature selection, multicollinearity handling and improved interpretability. The equation is as follows:

$$
Loss = MSE + Penalty = \frac{1}{n} \sum_{i=1}^{n} {(Y_i - \hat{Y}_i)^2} + \lambda \sum\_{j=1}^{n} |\beta_j|
$$

Moving away from matrix notation for lasso, lets go back to our formulation for linear regression. We aim to model the relationship between independent variables (features) and the dependent variable (target) as:

$$
y_i = \sum_{j=1}^p X_{ij}\beta_j + \epsilon_i
$$

Where:

- $y_i$ is the actual value of the dependent variable for the i-th observation.
- $X_{ij}$ is the value of the j-th feature for the i-th observation.
- $\beta_j$ is the coefficient for the j-th feature.
- $\epsilon_i$ is the true error. It represents the part of the data that the model cannot explain. This is often referred to as the random error or noise in the data.

We will have loss:

$$
loss = \sum\_{i=1}^n r_i^2 = \sum\_{i=1}^n (y_i-\hat{y_i})^2 = \sum\_{i=1}^n (y_i - \sum_{j=1}^p X_{ij}\beta_j)^2
$$

and correspondingly, objective function:

$$
\underset{\beta}{\min} \ \frac{1}{2n} \sum_{i=1}^n (y_i - \sum_{j=1}^p X_{ij}\beta_j)^2 + \lambda \sum_{j=1}^p |\beta_j|
$$

In this case we scale by n in order to remove the dependency on the size of the dataset.

##### Connecting Back

How does lasso regression help with multicollinearity, feature selection and interpretability? The penalty shrinks less important coefficients to exactly zero. For highly correlated features, Lasso tends to select one feature and discard others (setting their coefficients to zero). Lasso also reduces beta coefficients to 0, effectively removing redundant features and noise in the data. Of course, since there are fewer features understanding how independent variables correlate to our target will be easier, making our model more interpretable.

## Tree Architectures

### Basic Decision Trees

A decision tree partitions or splits the feature space into regions and predicts a constant value for each region. For regression this constant is the mean of the target variable in the chosen region.

Lets say we begin with some dataset $D = \{(x_i, y_i)\}_{i=1}^N$ where $x_i$ is a feature vector and $y_i$ is the target value. Taking the example of house prices, for a dataset with features $[\text{size, bedrooms, location, year built}]$, a single row of $x_i$ would be $[2000,3,\text{suburb},1995]$. The target value $y_i$ in this case would be the price of the house $[500000]$.

For regression, decision  trees use a weighted average of variances in order to find the optimal split. The weightage accounts for the size of splits. If one subset is very large compared to the other, it prevents small subsets from disproportionately affecting the evaluation of each split. This is given by:

$$
\text{Weighted Variance} = \frac{|D_L|}{D}\sigma^2(D_L) + \frac{|D_R|}{D}\sigma^2(D_R)
$$

where:

- $|D_L|$, $|D_R|$ is the number of samples in the left and right subset respectively.
- $|D|$ is the total number of samples in the parent node.
- $\sigma^2(D_L)$, $\sigma^2(D_R)$ is the variance of target values in the left and right subset respectively.

The question now is, how does the model decide which splits to evaluate? The  model, for each feature $x_j$, evaluates all feasible split points, called thresholds $t$. These thresholds are finite, determined based on the unique values of $x_j$. Lets take an example of $x_j = [100,200,300,400]$. The candidate thresholds are the midpoints between consecutive unique values in the sorted list:

$$
t \in \lbrace{\frac{100+200}{2}, \frac{200+300}{2}, \frac{300+400}{2}}\rbrace = \lbrace{150, 250, 350\}\rbrace
$$

For each threshold $t$ the algorithm evaluates the split by dividing the data into two groups:

 - $D_L = \lbrace{(x_i,y_i | x_{ij} < t}\rbrace$
 - $D_R = \lbrace{(x_i,y_i | x_{ij} \geq t}\rbrace$

For continous values, this is chosen discretely.

The stop this splitting process when one of the following scenarios occours:

- All target values are identical: If all target values are identical, the variance is by definition 0. Further splits cannot reduce it.
- Node contains only one data point: A single datapoint has no variance by defenition
- Thresholds Exhausted: If all possible splits for all features have been evaluated, and none result in a variance reduction.
- Predefined Stopping Criteria: This could be that the tree reaches a maximum depth, defined by the user of minimum samples per leaf.

### Random Forests

A random forest is an ensemble of decision trees, where each tree is built independently on a random subset of the data and features. The final prediction is made by aggregating the predictions of all trees. In the case of regression, the final prediction is the average of the predictions from all trees. For classification the final prediction is the majority vote.

In order to introduce diversity among the trees, each tree in the random forest is trained on a bootstrap sample of the dataset. What this means is that some instances of the dataset may appear multiple times while other may not appear at all. It creates $N$ instances with replacement from the original dataset.

To introduce further diversity, each tree does not consider all features of the input matrix $x_i$, instead only a random subset of $m$ features is chosen, where $m < p$ where $p$ is the total amount of features. 

The mathematical notation for the ensemble is given by:

$$
\hat{y}(x) = \frac{1}{B} \sum_{b=1}^B \hat{y}_b(x)
$$

for regression, and 

$$
\hat{y}(x) = \text{mode}(\lbrace{\hat{y}_1(x), \hat{y}_2(x), ... , \hat{y}_B(x)}\rbrace)
$$

for classification.

### XGBoost

Unlike bagging (random forests), boosting builds trees sequentially, each one correcting the errors of the previous trees using gradient-based optimization. 

XGBoost minimizes a regularized objective function:

$$
\text{Loss} = \sum_{i=1}^N \ell(y_i, \hat{y}\_i) + \sum_{k=1}^K \Omega(f_k)
$$

where:

- $\ell(y_i, \hat{y}\_i)$ is our loss function, which for regression could be a MSE, or MAE. In classification this would be a log loss.
- $\Omega(f_k)$ is the regularization term used in XGboost. It is given by:

$$
\Omega(f_k)  = \gamma T + \frac{\lambda}{2} \sum_{j=1}^T w_j^2
$$

Minimizing this loss function directly is complex and computationally expensive, especially when loss is non-linear, for example in the case of classifcation when we use a log-loss. To simplify this problem, we use a second-order taylor approximation to approximate the loss function locally around the current predictions $\hat{y}_i^{(t-1)}$. We then get for a small change $f_t(x)$, the taylor expansion of $\ell(y_i, \hat{y}_i^{(t)})$ around $\hat{y}_i^{(t-1)}$ is:

$$
f(x) \approx f(a) + f'(a)(x - a) + \frac{f''(a)}{2}(x - a)^2
$$

$$
\ell(y_i, \hat{y}_i^{(t)}) \approx \ell(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2
$$

where:
- $g_i = \frac{\partial \ell(y_i, \hat{y}_i)}{\partial \hat{y}_i}$ is the gradient which measures how the loss changes with $\hat{y}_i$.
- $h_i = \frac{\partial^2 \ell(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}$ is the hessian which measures how the gradient itself changes.

The information gain for XGBoost is given by:

$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H\_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$

So now that we know all this information lets put it all together: 

- At iteration $t-1$ the model predicts $\hat{y}_i^{(t-1)}$.
- It calculates gradients and hessians $g_i$, and $h_i$, for each data point based on the current predictions.
- The new tree $f_t(x)$ is trained to predict $g_i$, reducing the residual error, by learning how to correct the model's current errors.
- For each candidate split, we compute the gain using $G_L, G_R, H_L, H_R$ to find the best split.
- Adjust leaf weights in order to adjust predictions.
- Repeat for the desired number of iterations.
  
# Algorithm Complexity Note

Lets analyse the complexity of the models we have just seen. 

## Linear Regression

## Trees

### Basic Decision Tree

### Random Forest

### XGBoost

XGBoost's time complexity is given by:

$$
\text{complexity} = O(K \cdot d \cdot T \cdot n)
$$

Where :
- **K**: Number of estimators/trees
- **d**: Maximum depth per tree
- **T**: Number of non-zero rows
- **n**: Total number of train rows

For each node, for each feature, the algorithm finds the optimal split point by finding maximum gain. This does this for all n train rows. It may use some sorting feature in order to speed up the process. This is applied for K trees each with depth 5.


# Adding Temporal Dependencies using Deterministic Processes

## Linear Regression

For some models, like linear regression and decision trees, they are not able to understand temporal dependencies naturally. This is inherent to the way they work. Lets go over why they dont work.

Linear regression is a model that assumes that all data is independent and identically distriuted (i.i.d). It models the relationship between input features and the output as:


$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \epsilon
$$

This does not account for the ordering of the data, as it fails the i.i.d assumption. Linear regressors cannot model this time dependent relationship between past outputs and future outputs unless we explicitly state them!

### Decision Trees  

Trees make decisions by comparing features' values at each split. They do not consider sequential relationships between rows or recognize time as a continuous flow.

Tree-based models view time simply as another feature (e.g., 1, 2, 3, ...), without understanding that the data is ordered in time. Trends (upward/downward slopes) or periodicities exist.

When using deterministic processes, ensure that any order greater than 1 is strictly used for in sample modelling, otherwise we get exploding predictions.

## Deterministic Processes

Deterministic processes (e.g., linear trends, Fourier terms) encode temporal structure into the input features such that non-temporal models can understand temporal dependencies!

A deterministic process does this in 3 different ways: 

- Adding a trend term that explicitly models linear changes over time:


$$
trend = \beta x
$$

- Encodes seasonality:

$$
\sin(\frac{2\pi t}{P}) \quad \text{and} \quad \cos(\frac{2\pi t}{P})
$$

- Adding lag features (e.g $y_{t-n}$) to help the model understand how the past outputs relate to the current one.


For complex non-linear patterns, either try to introduce seasonality using fourier terms, or implement non-linear models like trees or neural networks.


### Introducing Seasonality

Analysing a periodogram in your data can be incredibly useful to indentify periodicities. In other words, in our data, what kind of seasonality is strongest? From our data in store sales, we can see that weekly seasonality has the highest variance, with a noticeable secondary peak at bi-weekly seasonality. 

Now that we know this, how can we give the model this information? There are 2 ways to do this:

- One hot Encoding
- Fourier Calendar Terms

#### One Hot Encoding
One hot encoding creates dummy binary (0/1) variables for each category within the period (e.g., day-of-week, month, hour). This type is better for more granular or short-term seasonality, where the changes are sharp and distinct. Some examples include: 
- Day-of-Week Seasonality: Sales spike on weekends but drop on weekdays.
- Hour-of-Day Seasonality: Traffic peaks during morning and evening rush hours.

The discrete binary values explicitly models sharp changes between categories without assuming continuity.

#### Fourier Calendar Terms

Fourier Calendar Terms represent seasonality using smooth sinusoidal functions (sine and cosine waves). These are better for larger or long-term seasonality where the shifts occur gradually and periodically. Some examples include:
- Annual Seasonality: Temperature across the year.
- Monthly Seasonality: Sales across months.

Fourier terms approximate patterns as smooth waves because sine and cosine functions are continuous. They do not model sharp or abrupt changes.

