import pandas as pd
import numpy as np
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

train = pd.read_csv('data/train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
        'onpromotion': 'uint32'
    },
    parse_dates=['date'],
)

train['date'] = train.date.dt.to_period('D')

test = pd.read_csv('data/test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
)
test['date'] = test.date.dt.to_period('D')
test_copy = test.copy()
test = test.set_index(['date', 'family', 'store_nbr'])


X_train = train.set_index(['date', 'family', 'store_nbr'])
X_train = X_train.unstack(['family', 'store_nbr'])
index_ = X_train.index.get_level_values('date').unique() 

y_train = X_train['sales']

dp = DeterministicProcess(
    index = index_,
    constant = True,
    order = 1,
    drop = True,                  
)

X_deterministic_train = dp.in_sample()
X_deterministic_test = dp.out_of_sample(steps=16)

regressor = LinearRegression()
regressor.fit(X_deterministic_train, y_train)

preds = regressor.predict(X_deterministic_test)
y_submit = pd.DataFrame(preds,               
                        index = X_deterministic_test.index,      
                        columns = y_train.columns).clip(0.0)

submit_df = (
    y_submit.stack(['store_nbr', 'family'], future_stack=True)
    .to_frame(name='sales')
    .reset_index()
    .rename(columns={'level_0': 'date'})
    .merge(test_copy, on=['date', 'store_nbr', 'family'], how='left')
)

submit_df[['id', 'sales']].to_csv('data/submissions/linreg_submission.csv', index=False)
