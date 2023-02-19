#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import torch
import numpy as np


# In[2]:


# creating a tensor
x = torch.arange(4.0, requires_grad=True)
x


# In[3]:


# the results
print(x.grad) # # The gradient is None by default


# In[4]:


# our optimization function
y = 2 * torch.dot(x, x)  # 2*14
y


# In[5]:


# calculating the gradient
y.backward()


# In[6]:


# the results
x.grad


# In[7]:


# the gradient of our function is 4x
4 * x


# In[8]:


# checking the automatic gradient computation  
x.grad == 4 * x


# In[9]:


# Reset the gradient
x.grad.zero_()


# In[10]:


# our 2nd optimization function
y = x.sum()
y


# In[11]:


# calculating the gradient
y.backward()
x.grad


# In[12]:


# Reset the gradient
x.grad.zero_()
# our function x^2
y = x * x
y


# In[13]:


# calculating the gradient
y.backward(gradient=torch.ones(len(y)))    # or y.sum().backward()
x.grad


# In[14]:


x.grad.zero_()
y = x * x
# Detaching y from the Computation
u = y.detach()
z = u * x
z.sum().backward()
x.grad 


# In[15]:


# even we detached y from the computational graph of z but we still have the computational graph of y
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x

