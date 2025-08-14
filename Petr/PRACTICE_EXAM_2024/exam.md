---
title: "My Exam"
fontsize: 12pt
geometry: margin=1.5cm
listings: true
---

<!---
######################################################

📌 Python code block:
```python

```
######################################################

📌  Figure using \fig macro (full width by default in YAML):
\fig{ }{X}



📌 Aray print precision:
print(’w_MLE’, np.array2string(w_MLE, precision=2)
-->

# Task 1: 

## 1.1 



\fig{image.png}{MLE and Beta OLS estimate}

**CODE**

```python
x = jnp.array([
    2.29, -1.8, -0.06, 3.72, 2.6, -5.93, -0.15
])
y = jnp.array([
    3.17, -4.53, -0.78, 3.15, 4.76, -1.96, -1.32
]).reshape(-1, 1)

def design_matrix(x):
    return jnp.column_stack((jnp.ones(len(x)), x**2, jnp.sin(x), x))

PHI = design_matrix(x)
  

model = BR.BayesianLinearRegression(Phi=PHI, y=y, beta=1, alpha=1)

w_MLE = model.get_w_mle()
print(f"The MLE of W is \n{w_MLE}")
The MLE of W is 
[[-0.73483423]
 [ 0.11247988]
 [ 2.35754774]
 [ 1.01247728]]

sigma = model.get_sigma2_mle()
print(f"The estimate of sigma is {sigma}")
print(f"We obtain beta by 1/sigma² = {1/sigma}")
The estimate of sigma is 0.20655746802181824
We obtain beta by 1/sigma² = 4.841267709063765
```

## 1.2 

\fig{image-2.png}{Post. pred. dist. mean, std and CI}


```python
x_star = jnp.array([1])

Phi_star = design_matrix(x_star)
print(Phi_star)
print(f"Phi_star_shape {Phi_star.shape}")


mean_plug_in = Phi_star @ w_MLE
print(f"The mean of post. pred. dist p(y*|y, x*=1) = {mean_plug_in.item():.2f}")

mean_plug_in = Phi_star @ w_MLE

print(f"The mean of post. pred. dist p(y*|y, x*=1) = {mean_plug_in.item():.2f}")

print(f"Variance is just {sigma:.2f}")
print(f"STD is {jnp.sqrt(sigma):.2f}")
print(f"Credibility intervals are {GCI.gaussian_credibility_interval(mean_plug_in, sigma)}")
```

# Task 1:
 
# Task 1:

# Task 1:

# Task 1:
