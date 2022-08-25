[![](https://github.com/amirhm/embedml/actions/workflows/build.yml/badge.svg)](https://github.com/amirhm/embedml/actions/workflows/build.yml)


# embedML

pytorch like machine learning framework from scratch
I started the repo with building a simple compiler for embeded platfroms, but later decided to make a self contained abstract baby machine learning framework. 

Could be used mostly for the Educational purpose as well. Easy and full python implemtation using numpy. 
Smaller sibling or framworks like pytorch or jax :) and useful to see how the behind the scene of those bigger platfrom.

***and it is only less than 200 lines of code!***

## Instal

````
pip install embedml
````


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

# Example of differentiation of a funtion 
form Jax example:
```python
from jax import grad
import jax.numpy as jnp

def tanh(x):  # Define a function
  y = jnp.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)  # Obtain its gradient function
print(grad_tanh(1.0))   # Evaluate it at x = 1.0
# prints 0.4199743
```

and similar (backward gradient) with embedML:
```python

import numpy as np
from embedml.tensor import Tensor

def tanh(x):  # Define a function
    y = (-2 * x).exp()
    return (1 - y).div((1 + y))

x = Tensor(np.array(1))
y = tanh(x)
y.backward()

print(x.grad)
```
