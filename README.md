Lessons Learned:

Always time index your data! Without time indexing the model has no idea how different features relate. You essentially just pass 1 large input vector  (time seres) into your data. 

Ensure that before you even start looking at model architectures, your data is time indexed.

When using deterministic processes, ensure that any order greater than 1 is strictly used for in sample modelling, otherwise we get exploding predictions. At times we may use this for some applications but the scope must be limited, and analysis rigorous.

For complex non-linear patterns, either try to introduce seasonality using fourier terms, or implement non-linear models like trees or neural networks.

