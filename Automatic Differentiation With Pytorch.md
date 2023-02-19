```python
# import libraries
import torch
import numpy as np
```


```python
# creating a tensor
x = torch.arange(4.0, requires_grad=True)
x
```




    tensor([0., 1., 2., 3.], requires_grad=True)




```python
# the results
print(x.grad) # # The gradient is None by default
```

    None
    


```python
# our optimization function
y = 2 * torch.dot(x, x)  # 2*14
y
```




    tensor(28., grad_fn=<MulBackward0>)




```python
# calculating the gradient
y.backward()
```


```python
# the results
x.grad
```




    tensor([ 0.,  4.,  8., 12.])




```python
# the gradient of our function is 4x
4 * x
```




    tensor([ 0.,  4.,  8., 12.], grad_fn=<MulBackward0>)




```python
# checking the automatic gradient computation  
x.grad == 4 * x
```




    tensor([True, True, True, True])




```python
# Reset the gradient
x.grad.zero_()
```




    tensor([0., 0., 0., 0.])




```python
# our 2nd optimization function
y = x.sum()
y
```




    tensor(6., grad_fn=<SumBackward0>)




```python
# calculating the gradient
y.backward()
x.grad
```




    tensor([1., 1., 1., 1.])




```python
# Reset the gradient
x.grad.zero_()
# our function x^2
y = x * x
y
```




    tensor([0., 1., 4., 9.], grad_fn=<MulBackward0>)




```python
# calculating the gradient
y.backward(gradient=torch.ones(len(y)))    # or y.sum().backward()
x.grad
```




    tensor([0., 2., 4., 6.])




```python
x.grad.zero_()
y = x * x
# Detaching y from the Computation
u = y.detach()
z = u * x
z.sum().backward()
x.grad 
```




    tensor([0., 1., 4., 9.])




```python
# even we detached y from the computational graph of z but we still have the computational graph of y
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```




    tensor([True, True, True, True])


