![Banner](assets/banner_time_series.png)
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
- No Multicollinearity: Independent variables ($X_1, X_2, ... , X_n) should not be highly correlated with each other. When 2 or more independent variables are highly correlated, they carry redundant information, which makes it hard to determine their individual effect on the target variable $Y$. We can detect multicollinearity by calculating a ocrrelation matrix or calculating the variance inflation factor.

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
RSS = (Y - X\beta)^T (Y - X\beta)
$$

Taking the derivative of RSS with respect to $\beta$ and setting it to zero gives:

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

We can avoid overfitting and multicollinearity in our linear regression model by using Ridge (L2) or Lasso (L1) Regularization. How it works is by adding penalties at the end of our loss function:

$$
Loss = MSE + Penalty = \frac{1}{n} \sum_{i=1}^{n} {(Y_i - \hat{Y}_i)^2} + \lambda \sum \beta^2
$$

Lets dive deeper into how the linear algebra here. From our equation of OLS:

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

Lets consider an example to make this more clear visually. Consider design matrix $X#:

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
\end{bmatrix}
$$


#### Lasso (L1)



## Tree Architectures

### 


# Adding Temporal Dependencies using Deterministic Processes

### Linear Regression

For some models, like linear regression and decision trees, they are not able to understand temporal dependencies naturally. This is inherent to the way they work. Lets go over why they dont work.

Linear regression is a model that assumes that all data is independent and identically distriuted (i.i.d). It models the relationship between input features and the output as:


$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \epsilon
$$

This does not account for the ordering of the data, as it fails the i.i.d assumption. Linear regressors cannot model this time dependent relationship between past outputs and future outputs unless we explicitly state them!

### Decision Trees  

Tree-based models (e.g., Decision Trees, Random Forests, XGBoost) split the data into regions by evaluating individual features independently. Trees make decisions by comparing features' values at each split. They do not consider sequential relationships between rows or recognize time as a continuous flow.

Tree-based models view time simply as another feature (e.g., 1, 2, 3, ...), without understanding that the data is ordered in time. Trends (upward/downward slopes) or periodicities exist.

When using deterministic processes, ensure that any order greater than 1 is strictly used for in sample modelling, otherwise we get exploding predictions. At times we may use this for some applications but the scope must be limited, and analysis rigorous.

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

# Algorithm Complexity Note

Lets take a moment to analyse the complexity of our models. 

## Linear Regression


## XGBoost

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
