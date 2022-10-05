# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python [conda env:min_prec]
#     language: python
#     name: conda-env-min_prec-py
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# # Comments on "Optimizing Early Warning Classifiers to Control False Alarms via a Minimum Precision Constraint"
#
#  by Preetish Rath, Michael C. Hughes
#  
#  Commentary by Jason Myers and Ilan Fridman Rojas

# + [markdown] slideshow={"slide_type": "slide"}
# ## Introduction
#
# There are many applications where *alarm fatigue* is a serious operational issue.
#
# False positives lead to exhaustion, and to users ignoring predicted positives

# + [markdown] slideshow={"slide_type": "slide"}
# INGA WB has had at least two products which revolve around this issue:
#     
# - *Hunter*: rule-based transaction monitoring generated so many positives that investigators were permanently overwhelmed
# - *CodeFix*: false positives from static analysis code vulnerability tools mean software engineers are swamped with pointless alerts and might miss serious issues

# + [markdown] slideshow={"slide_type": "slide"}
# The solution and goal: reduce false positives
#
# Or equivalently: increase/control precision
#
# $$p = \frac{\mathrm{tpc}}{\mathrm{tpc}+\mathrm{fpc}}$$

# + [markdown] slideshow={"slide_type": "slide"}
# How this problem is usually tackled:
# 1) predict probabilities on validation set  
# 2) threshold the probabilities at a range of different thresholds and compute precision and recall  
# 3) draw the precision-recall curve  
# 4) choose a threshold which satisfies requirements (if one exists

# + [markdown] slideshow={"slide_type": "slide"}
# This has the disadvantage that the precision-recall constraints are superimposed on the problem... they are not part of the learning problem
#
# Therefore one can imagine that, for appropriate and purpose-built losses, one could find a better fit/decision boundary (in terms of precision and recall requirements)
#
# That is exactly what this paper claims, demonstrates in principle (on a synthetic example), and shows one way to construct such a loss
#
# "(...) our specific goal is to maximize recall subject to a precision of at least 0.9. The BCE solution gets only 0.69 precision even with post-hoc threshold search (...) our approach can satisfy the desired precision constraint."
#
# (BCE = Binary Cross-Entropy, i.e. logistic regression)

# + [markdown] slideshow={"slide_type": "slide"}
# The goal is conceptually clear
#
# But how could one actually formulate such a model/loss, and what should it aim for?
#
# Maximise precision regardless of recall?
# Maximise precision for a minimum targetted recall?
# Maximise recall for a minimum targetted precision?
#
# This will be context-dependent. An example will clarify

# + [markdown] slideshow={"slide_type": "slide"}
# Consider use case where:
# - Events are rare
# - A single event is very costly
# - Interventions are a limited resource
#
# The goal is to predict events, so interventions can be targeted to prevent the actual event

# + [markdown] slideshow={"slide_type": "slide"}
# Concretely, consider the medical context (the example they give).
#
# False positives can have a high cost:
# - A false positive results in someone getting unnecessarily treated (possibly harmed)
# - Someone else (a true positive) possibly missing out on treatment if it was used elsewhere
# - Medical staff ignoring predicted positives since they are too often false, therefore more people suffering harm or death
#
# Note: this scenario holds for transaction monitoring (Hunter) or code vulnerability scenarios (CodeFix) as well

# + [markdown] slideshow={"slide_type": "slide"}
# Therefore targetting a _minimum precision_ is highly desirable in these scenarios.
#
# At the same time, we do not want to miss cases which require intervention, so _recall should be maximised as well_.

# + [markdown] slideshow={"slide_type": "slide"}
# The loss can therefore be schematically written as 
#
# $$\max \mathrm{tpc} \;\;\;\;\mathrm{subject\, to}\;\; \frac{\mathrm{tpc}}{\mathrm{tpc} + \mathrm{fpc}} \geq \alpha$$
#
# where $\alpha$ is the minimum targeted precision  
#
# (Remember that recall is the true positive count divided by the total number of positive examples in the dataset: $r = \frac{\mathrm{tpc}}{N_+}$. But $N_+$ isn't something we control, so maximising recall is the same as maximising $\mathrm{tpc}$.)

# + [markdown] slideshow={"slide_type": "slide"}
# This can be rewritten as
#
# $$\min -\mathrm{tpc} \;\;\;\;\mathrm{subject\, to}\;\; \underbrace{-\mathrm{tpc} + \frac{\alpha}{1-\alpha} \mathrm{fpc} }_{g} \leq 0$$
#
# (but to derive this we had to assume $\mathrm{tpc} + \mathrm{fpc} > 0$, we will have to be careful to respect this at all times later)

# + [markdown] slideshow={"slide_type": "slide"}
# Transforming the constrained optimisation problem into an unconstrained one using the penalty method
#
# $$\min \left[ -\mathrm{tpc} + \lambda \max(0,\, g) \right]$$
#
# Note that whenever $g > 0$ this imposes a penalty on the loss, as expected.  $\lambda$ is a penalty parameter (set to 1000 or 10000 by default), it controls the steepness of loss imposed when constraint $g \leq 0$ is violated.
#
# **This is the optimisation problem we would like to solve**

# + [markdown] slideshow={"slide_type": "slide"}
# But how would one calculate tpc and fpc?
#
# The true positive count, $\mathrm{tpc}$, is given by
# $$\mathrm{tpc} = \sum_{i: y_i = 1} \hat{y}$$
#
# Consider a learning algorithm, $f_\theta(x)$, which produces a score. This score is compared against a threshold, $b$, to produce a predicted class, i.e.
# $$\hat{y}_i = z[f_\theta(x_i) - b]$$
# where $z$ is a Heaviside step function

# + [markdown] slideshow={"slide_type": "slide"}
# We then have 
# $$\mathrm{tpc} = \sum_{i: y_i = 1} z[f_\theta(x_i) - b]$$
#
# ($\mathrm{fpc}$ is defined analogously: $\mathrm{fpc} = \sum_{i: y_i = 0} z[f_\theta(x_i) - b]$)

# + slideshow={"slide_type": "-"}
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 100)
y = np.where(x<0, 0, 1)

plt.plot(x, y, drawstyle='steps', linestyle='dotted')
plt.xlabel(r'$f_\theta(x)-b$')
plt.ylabel(r'$\hat{y}$')
plt.show()

# + [markdown] slideshow={"slide_type": "slide"}
# We would like to make the thresholding part of the learning algorithm (i.e. maximise $\mathrm{tpc}$)
#
# But the Heaviside step function is non-differentiable (at zero) and has zero gradient elsewhere, so gradient-based optimisation can't learn through it
#
# (And gradient-free optimisation methods don't scale well to many dimensions)  
#
# Therefore we need a differentiable approximation to the step function...

# + [markdown] slideshow={"slide_type": "slide"}
# But there is a noteworthy complication...
#
# Rewrite the minimum precision constraint as
# $$(1-\alpha)\, \mathrm{tpc} \;\geq\; \alpha \,\mathrm{fpc} $$
#
# The step-function $\mathrm{tpc}$ and $\mathrm{tpc}$ must satisfy this constraint... _and their approximation must satisfy it too!_  
#
# A single, differentiable approximation to the step function might not satisfy this

# + [markdown] slideshow={"slide_type": "slide"}
# Therefore we would like to _lower bound_ $\mathrm{tpc}$ and _upper bound_ $\mathrm{fpc}$  
# (See Lemma 4.1 in (Eban, 2017) for proof that this also satisfies the constraint)
#
# We would therefore like to define two, _differentiable_ functions which sandwich (lower bound and upper bound) the step function
#
# Use sigmoids!
# $$\sigma(x) = \frac{1}{1+e^{-x}}$$

# + [markdown] slideshow={"slide_type": "slide"}
# We would therefore like to define two, _differentiable_ functions which sandwich (lower bound and upper bound) the step function
#
# Use sigmoids!
# $$\sigma(x) = \frac{1}{1+e^{-x}}$$

# + [markdown] slideshow={"slide_type": "slide"}
# To upper bound the $\mathrm{fpc}$ we define a scaled sigmoid
#
# $$u(a) = (1 + \gamma\delta) \sigma(m a + b)$$
#
# where the $(1 + \gamma\delta)$ prefactor is to ensure the sigmoid upper bounds the maximum of the step function (with $\gamma \geq 1$, $\delta > 0$)

# + [markdown] slideshow={"slide_type": "slide"}
# We also require that the sigmoid rises sharply, so it is a tight bound around the step function
#
# In this case, ensure it goes from slightly above $0$ to slightly above $1$, in a short interval of length $\epsilon$
#
# $$\begin{align}
# u(-\epsilon) &\approx \delta \\
# u(0) &\approx 1+\delta
# \end{align}$$
#
# for $\epsilon > 0, \delta > 0$ but both small
#
# (too small would make the problem intractable again, as the sigmoid would become arbitrarily close to a step function again)

# + [markdown] slideshow={"slide_type": "slide"}
# Fixing $\gamma$, $\epsilon$ and $\delta$ to arbitrary but small values we can then optimise for the remaining two parameters of the sigmoid
#
# $$\hat{m},\,\hat{b} \;=\; \mathop{\mathrm{arg\,min}}_{m\in\mathbb{R},\, b\in\mathbb{R}}  \left[(u(-\epsilon) - \delta)^2 + (u(0)-(1+\delta))^2 \right]$$
#
# This is non-convex. But one can find a stationary point in it using PyTorch

# + slideshow={"slide_type": "slide"}
import torch
import torch.nn as nn
import torch.optim as optim

class tight_sigmoid_upper(nn.Module):
    def __init__(self, eps, delta, gamma):
        super().__init__()
        
        self.eps = eps
        self.delta = delta
        self.gamma = gamma
        
        self.m = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        
    def forward(self):
        
        def u(a):
            return (1+self.gamma*self.delta)*torch.sigmoid(self.m*a + self.b)
        
        obj = torch.sum(torch.square(self.delta - u(-self.eps)) + torch.square(1 + self.delta - u(0)))
        return obj


# + slideshow={"slide_type": "skip"}
eps = 0.75
delta = 0.035
gamma = 7

model_upper = tight_sigmoid_upper(eps, delta, gamma)

optimizer_upper = optim.Adam(model_upper.parameters(), lr=1)

for _ in range(1000):
    optimizer_upper.zero_grad()
    loss = model_upper()
    loss.backward()
    optimizer_upper.step()

# + [markdown] slideshow={"slide_type": "slide"}
# This correctly reproduces the fitted values shown in the paper  
#
# "we use BFGS to solve the minimization problem and yield slope mˆ = 6.85 and shift ˆb = 1.59"

# + slideshow={"slide_type": "-"}
m_upper, b_upper = model_upper.named_parameters()
m_upper = float(m_upper[1].detach().numpy())
b_upper = float(b_upper[1].detach().numpy())

print((m_upper, b_upper))
# + [markdown] slideshow={"slide_type": "slide"}
# Now let's lower bound $\mathrm{tpc}$  
#
# Roadblock: the sigmoid would dip below zero, which could violate $\mathrm{tpc} + \mathrm{fpc} > 0$
#
# Trick: shift $\mathrm{tpc}$ step function up so sigmoid below it is always non-negative
#
# Make sure this shift doesn't affect the optimisation problem's solution

# + [markdown] slideshow={"slide_type": "slide"}
# The step function for $\mathrm{tpc}$ is shifted up by $\gamma\delta N_+$ (where $N_+$ is the number of ground truth positive cases)
#
# $$\mathrm{tpc} \;\;\rightarrow\;\; \mathrm{tpc} + \gamma\delta N_+$$
#
# An equal amount is added to the penalty so that the constraint is unaltered (since $g$ also contains a $-\mathrm{tpc}$ term in its definition)
#
# $$g \;\;\rightarrow\;\; g + \gamma\delta N_+$$
#
# The overall shifted optimisation problem is then
#
# $$\min \left[ -\mathrm{tpc} + \lambda \max(0,\, g) \right] \;\;\rightarrow\;\; \min \left[ -\mathrm{tpc}\, \underbrace{-\gamma\delta N_+}_{\mathrm{new}} + \lambda \max(0,\, g) \right]$$

# + [markdown] slideshow={"slide_type": "slide"}
# Lemma 4.1 in paper claims this modification does not change solution to optimisation problem  
# **I think this is not quite correct**    
#
# The constraint is unchanged by the shift, but the objective function is shifted, and $\min \mathrm{tpc} \neq \min \mathrm{tpc}+c$, for some constant c  
#
# (Argmin would be invariant to an additive constant, but min is not)  

# + [markdown] slideshow={"slide_type": "slide"}
# For example, think of $f(x)=x^2$  
# Then $\arg\min f(x) = \arg\min f(x)+c = 0$  
# But $\min f(x) = 0$ and $\min f(x)+c = c$, therefore $\min f(x) \neq \min f(x)+c$
#
# But any positive affine (or monotone increasing) function of $\mathrm{tpc}$ would work fine, since it still maximises $\mathrm{tpc}$  
# The actual final minimum value obtained doesn't matter, only that we optimised for increasing $\mathrm{tpc}$  
#
# Numerically, as long as $\gamma\delta \ll 1$ and $N_+$ is not too large, such that $\gamma\delta N_+ \approx 0$, then this difference won't matter  
#
# So this is fine nonetheless, I think  

# + [markdown] slideshow={"slide_type": "slide"}
# For the lower bound we use the same functional form as for the upper bound
#
# $$l(a) = (1 + \gamma\delta) \sigma(m a + b)$$
#
# but we require
#
# $$\begin{align}
# l(0) &\approx \tilde{\delta} \\
# l(+\tilde{\epsilon}) &\approx 1+\tilde{\delta}
# \end{align}$$
#
# This gives the optimisation problem
#
# $$\hat{m},\,\hat{b} \;=\; \mathop{\mathrm{arg\,min}}_{m\in\mathbb{R},\, b\in\mathbb{R}}  \left[(l(0) - \delta)^2 + (l(+\epsilon)-(1+\delta))^2 \right]$$

# + [markdown] slideshow={"slide_type": "slide"}
# Note that $\tilde{\epsilon},\, \tilde{\delta}$ don't necessarily have any relation to the $\epsilon,\, \delta$ used before  
#
# However for convenience we will actually use the same values: $\tilde{\epsilon}=\epsilon,\, \tilde{\delta}=\delta$  
#
# Also note that to ensure the upper bound on $\mathrm{fpc}$ and lower bound on $\mathrm{tpc}$ do not intersect each other, but are as tight as possible, we require them to asymptote to the same values:
#
# $$\begin{align}
# u(-\infty) = l(-\infty) &= 0 \\
# u(\infty) = l(\infty) &= 1+\gamma\delta
# \end{align}$$

# + slideshow={"slide_type": "slide"}
import torch
import torch.nn as nn
import torch.optim as optim

class tight_sigmoid_lower(nn.Module):
    def __init__(self, eps, delta, gamma):
        super().__init__()
        
        self.eps = eps
        self.delta = delta
        self.gamma = gamma
        
        self.m = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        
    def forward(self):
        
        def u(a):
            return (1+self.gamma*self.delta)*torch.sigmoid(self.m*a + self.b)
        
        obj = torch.sum(torch.square(self.delta - u(0)) + torch.square(1 + self.delta - u(self.eps)))
        return obj


# + slideshow={"slide_type": "skip"}

model_lower = tight_sigmoid_lower(eps, delta, gamma)

optimizer_lower = optim.Adam(model_lower.parameters(), lr=1)

for _ in range(1000):
    optimizer_lower.zero_grad()
    loss = model_lower()
    loss.backward()
    optimizer_lower.step()

# + [markdown] slideshow={"slide_type": "slide"}
# "Concretely fixing tolerances to γ = 7.00, δ=0.035, ε=0.75, we obtain m ̃=6.85 and  ̃b=−3.54"

# + slideshow={"slide_type": "-"}
m_lower, b_lower = model_lower.named_parameters()
m_lower = float(m_lower[1].detach().numpy())
b_lower = float(b_lower[1].detach().numpy())

print((m_lower, b_lower))
# +
from scipy.special import expit

def bound_sigmoid(x, m=1, b=0, eps=0.75, delta=0.035, gamma=7.):
    return (1+gamma*delta)*expit(m*x + b)


# + slideshow={"slide_type": "-"}
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 100)
y = np.where(x<0, 0, 1) + gamma*delta
y_bound = bound_sigmoid(x, m=m_lower, b=b_lower)

plt.title('tpc')
plt.plot(x, y, drawstyle='steps', linestyle='dotted')
plt.plot(x, y_bound)
plt.xlabel(r'$f_\theta(x)-b$')
plt.ylabel(r'$\hat{y}$')
plt.show()

# + slideshow={"slide_type": "-"}
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 100)
y = np.where(x<0, 0, 1)
y_bound = bound_sigmoid(x, m=m_upper, b=b_upper)

plt.title('fpc')
plt.plot(x, y, drawstyle='steps', linestyle='dotted')
plt.plot(x, y_bound)
plt.xlabel(r'$f_\theta(x)-b$')
plt.ylabel(r'$\hat{y}$')
plt.show()

# + slideshow={"slide_type": "-"}
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 100)

y_tpc = np.where(x<0, 0, 1) + gamma*delta
y_tpc_bound = bound_sigmoid(x, m=m_lower, b=b_lower)

y_fpc = np.where(x<0, 0, 1)
y_fpc_bound = bound_sigmoid(x, m=m_upper, b=b_upper)

plt.plot(x, y_tpc, drawstyle='steps', linestyle='dotted', color='blue')
plt.plot(x, y_tpc_bound, color='blue', label='tpc')

plt.plot(x, y_fpc, drawstyle='steps', linestyle='dotted', color='green')
plt.plot(x, y_fpc_bound, color='green', label='fpc')

plt.xlabel(r'$f_\theta(x)-b$')
plt.ylabel(r'$\hat{y}$')

plt.legend()
plt.show()
# -

# We can now use these sigmoids in our optimisation problem, and we now have to minimise a sum of sigmoids, with a penalty term included too
#
# This is a difficult, non-convex problem  
# (Objective is non-convex, and we haven't checked whether our constraint forms a convex set)
#
# See e.g. https://web.stanford.edu/~boyd/papers/pdf/max_sum_sigmoids.pdf

# # loss landscapes
#
# Since we are looking at simple linear models we can actually take a look at the loss-landscape. It provides some useful insights.
#
# Since the loss function requires data to be defined we will use the dataset found in repo used by Rath and Hughes. We will discuss this in more detail at some point but this dataset is the type that the minimum precision loss seems to work best for.


# +
from data import toydata

x_toy, y_toy, _, _, _, _ = toydata.create_toy_dataset()

x = torch.from_numpy(x_toy).float()
y = torch.from_numpy(y_toy).float()

# +
import matplotlib.pyplot as plt

cdict = {0: 'green', 1: 'blue'}
mdict = {0:'o', 1:'x'}

fig, ax = plt.subplots()
for g in np.unique(y_toy):
    ix = np.where(y_toy == g)
    ax.scatter(x_toy[ix,0], x_toy[ix,1], c = cdict[g], marker=mdict[g], label = g, alpha=0.3)
ax.legend()

plt.show()

# +
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
import numpy as np

from constrained_metric_loss.min_precision_loss import MinPrecLoss

init_notebook_mode()

# +

sklearnlogreg = LogisticRegression()
sklearnlogreg = sklearnlogreg.fit(x,y)
sklearnbetas = np.concatenate([sklearnlogreg.coef_.flatten(), sklearnlogreg.intercept_])


# -

def bce_loss(beta):
    torch_beta = torch.from_numpy(beta).float()
    model_func = lambda xx:  xx @ torch_beta

    x_w_dummy_for_int = np.column_stack((x, np.ones([x.shape[0]])))
    x_w_dummy_for_int = torch.from_numpy(x_w_dummy_for_int).float()

    loss_funct = nn.BCEWithLogitsLoss()

    return loss_funct(model_func(x_w_dummy_for_int), y).numpy()

# + slideshow={"slide_type": "skip"}


def min_prec_loss(beta, min_prec=0.9, lmbda=100):
    torch_beta = torch.from_numpy(beta).float()
    model_func = lambda x:  x @ torch_beta

    x_w_dummy_for_int = np.column_stack((x, np.ones([x.shape[0]])))
    x_w_dummy_for_int = torch.from_numpy(x_w_dummy_for_int).float()
    
    f = model_func(x_w_dummy_for_int)
    
    loss = MinPrecLoss(
        min_prec = min_prec,
        lmbda = lmbda,
        sigmoid_hyperparams = {"gamma": 7, "delta": 0.035},
        sigmoid_params = {'mtilde': 6.85,'btilde': -3.54, 'mhat': 6.85, 'bhat': 1.59}
    )
    
    return loss.forward(f, y).numpy()


# -


def get_loss_landscape(loss_function, num_samples, w0_width, w1_width, kwargs={}):
    N = num_samples
    xv, yv = np.meshgrid(
        sklearnbetas[0] + np.linspace(-w0_width, w0_width, N), 
        sklearnbetas[1] + np.linspace(-w1_width, w1_width, N)
    )
    input_params = np.column_stack([xv.ravel(), yv.ravel(), sklearnbetas[2]*np.ones(N*N)])

    losses = np.apply_along_axis(loss_function, 1, input_params, **kwargs)
    
    data = [
        go.Surface(z=losses.reshape(xv.shape), x=xv, y=yv),
        go.Scatter3d(
            x = [sklearnbetas[0]], 
            y = [sklearnbetas[1]], 
            z = [loss_function(sklearnbetas, **kwargs)], 
            mode='markers',
            marker=dict(size=12, color='black')
        )
    ]

    layout = go.Layout(
        width=800,
        height=800,
    )
    return go.Figure(data=data, layout=layout)


# In the following plots the model's intercept value is held equal at the value obtained using BCE with logits loss. Then we vary the weights of the features for the x-y axis. The total loss value is plotted on the z-axis. 

iplot(get_loss_landscape(bce_loss, 40, 4, 4, {}))

iplot(get_loss_landscape(min_prec_loss, 40, 4, 4, {'min_prec': 0.9, 'lmbda': 1e3}))

# # Implementation and performance
#
# Due to the non-linear nature of the loss function we find that the initilization of the gradient decent is vital. In order to account for this the authors of the paper try a large range of random initial conditions and select the best one. 
#
# On the same toy dataset as we used above let's observe what decision boundaries are being drawn on the data. We will use the optimal boundary where we define optimal as one that most closely meets the minimum precision bound whilst maximsing the FBeta score on the test data.







# # Outlook
#
# An important realization in recent weeks is to try understand the role

# + [markdown] slideshow={"slide_type": "skip"}
# Put it all together: we have a loss function to fit!

# + [markdown] slideshow={"slide_type": "skip"}
# Show PyTorch implementation

# + [markdown] slideshow={"slide_type": "skip"}
# Show various fits on moons dataset with different minimum precisions: show that it works

# + [markdown] slideshow={"slide_type": "skip"}
# Discuss non-convexity and initialisation trick (fit log reg first)
#
# Quote what they say about non-convexity
#
# Their solution is to do many runs: computationally expensive
#
# Our solution: initialise using logistic regression
#
# Other, possibly better minima might exist

# + [markdown] slideshow={"slide_type": "skip"}
# PLOTS OF LOSS LANDSCAPE AND DISCUSSION OF IT HERE

# + [markdown] slideshow={"slide_type": "skip"}
# Next steps
#
# - Apply to a non-linear model (beyond logistic regression)
# - Check if our constraint forms a convex set (purely out of interest, our objective is non-convex so the problem is overall non-convex anyways)
# - End-to-end training, using e.g. a Lipschitz constraint to ensure sigmoids remain differentiable (instead of their ad hoc parameter tuning of m and b). See e.g. https://ai.stackexchange.com/a/29927
# - Courant-Beltrami improvement?
# - Make more parameters learnable (m and b, for tpc and fpc)
# - Add tpc+fpc>0 as constraint
# - Examine sensitivity to epsilon, delta, gamma
# - Other initialisation improvements
# - What is gained by tightness of bounds? How big a difference does it make?
# - Connection to Firth correction?
# - Something else?
# -


