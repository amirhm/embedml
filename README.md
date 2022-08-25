[![](https://github.com/amirhm/embedml/actions/workflows/build.yml/badge.svg)](https://github.com/amirhm/embedml/actions/workflows/build.yml)


# embedML

pytorch like machine learning framework from scratch
I started the repo with building a simple compiler for embeded platfroms, but later decided to make a self contained abstract baby machine learning framework. 

Could be used mostly for the Educational purpose as well. Easy and full python implemtation using numpy. 
Smaller sibling or framworks like pytorch or jax :) and useful to see how the behind the scene of those bigger platfrom.



## Example tarinig simiar to pytorch:

```python
y = model(x)
loss = criterion(y, t)
loss.backward()   
opt.step()
opt.zero_grad()
````

or like this implement the gradient decent:
```python
params = model.get_parameters()
for param in params:
  param -= param.grad * self.lr
```



