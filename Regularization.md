## Regularization

- Regularization is an umbrella term that encompasses methods that force the learning
algorithm to build a less complex model. In practice, that often leads to slightly higher
bias but significantly reduces the variance. This problem is known in the literature as the
bias-variancetradeoff


- The two most widely used types of regularization are called L1 and L2 regularization.The
idea is quite simple. To create a regularized model, we modify the objective function by
adding a penalizing term whose value is higher when the model is more complex.

![l1andl2.png](attachment:l1andl2.png)

- Lambda is a hyperparameter that controls the importance of regular-ization. If we set Lambda
to zero, the model becomes a standard non-regularized linear regression
model. On the other hand, if we set to
C
to a high value, the learning algorithm will try
to set most all the weights 
to a very small value or zero to minimize the objective, and the model will
become very simple which can lead to underfitting.


- Your job is to find lambda such that it's not underfitting but reduces the variance a little.


- In practice, L1 regularization produces a
**sparse model
, a model that has most of its
parameters equal to zero, provided the hyperparameter
lambda
is large enough. So L1 performs
feature selection
by deciding which features are essential
for prediction and which are not.** That can be useful in case you want to increase model
explainability. However, if your only goal is to maximize the performance of the model on
the holdout data, then L2 usually gives better results. L2 also has the advantage of **being
differentiable, so gradient descent can be used for optimizing the objective function.**
L1 and L2 regularization methods were also combined in what is called
elastic net regu-
larization
with L1 and L2 regularizations being special cases. You can find in the literature
the name
ridge regularization
for L2 and
lasso
for L1.


- In addition to being widely used with linear models, L1 and L2 regularization are also
frequently used with neural networks and many other types of models, which directly
minimize an objective function.

    Neural networks also benefit from two other regularization techniques:
    dropout
    and
    batch-
    normalization
    . There are also non-mathematical methods that have a regularization effect
    
![l1vsl2.png](attachment:l1vsl2.png)


```python

```
