
# Fundamental Step: Time Indexing

Always time index your data! Without time indexing the model has no idea how different features relate. You essentially just pass 1 large input vector  (time seres) into your data. Ensure that before you even start looking at model architectures, your data is time indexed.

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
