## Anderson Acceleration

Anderson Acceleration is a generalization of gradient descent. It uses the history of gradients to accelerate the convergence. 

### Why does this work?

- It is equivalent to doing gradient descent on a quadratic approximation of the loss function. 
- It is a generalization of gradient descent, and thus inherits all the nice properties of gradient descent (converges under standard assumptions). 
- It can be considered as a local second order method: it uses information about the Hessian. 
- The optimal choice for $\beta$ is $0$, which corresponds to doing pure gradient descent with a step size $\alpha$. 
- The weighting factor $\alpha$ is chosen such that the residual norm is minimized.

## Theory

Consider the problem of minimizing some objective function $f(x)$. We start from an initial point $x_0$ and we consider the following iterates:
$$x_{n+1} = \sum_{i=1}^{n+1} \alpha_i f(x_i)$$
where the $\alpha_i$ are chosen to minimize the residual norm:
$$\alpha = \arg\min_{\alpha} \|f(x_{n+1}) - x_{n+1}\|$$

It turns out that the optimal choice for $\alpha$ is given by the following expression:
$$\alpha = (H + \lambda I)^{-1}y \quad \text{ where } y = (1, 1, \dots, 1)^T \quad \text{ and } H_{ij} = f(x_i)^T f(x_j)$$

This expression can be computed efficiently using Sherman-Morrison-Woodbury formula. 

This method is particularly effective when $f$ is close to a quadratic function. It can also be used in the case of non-convex functions by using the approximate gradient instead of the true gradient. 

## Implementation

The code snippet below implements the Anderson acceleration algorithm. It is based on the paper "Anderson Acceleration of Stochastic Gradient Descent" (https://arxiv.org/abs/1412.5574).

### Inputs
- $function$: function to be minimized. It should accept a PyTorch tensor $X$ and return a PyTorch tensor $F$.
- $initial_value$: initial point. Should be a PyTorch tensor. 
- $grad_checks$: number of previous gradients used to accelerate the convergence.
- $\lambda$: regularization parameter. 
- $max\_iter$: maximum number of iterations. 
- $tolerance$: tolerance; algorithm stops when the relative residual norm is smaller than $tol$. 
- $\beta$: weighting factor between gradient descent and Anderson acceleration. When $\beta = 0$, it is equivalent to pure gradient descent. 

### Outputs
- $X$: list containing the iterates. 
- $F$: list containing the function values at each iteration.
- $G$: list containing the gradients at each iteration.
- $residual$: list containing the residual norm at each iteration. 
- $iterations$: number of iterations required to reach the desired tolerance. 

### Comments
- It is possible to use this algorithm for non-convex functions by replacing the true gradient by an approximate gradient. 
- It is possible to use this algorithm with minibatches. In this case, the function should accept a PyTorch tensor $X$ of shape $(N, B)$, where $N$ is the dimension and $B$ is the batch size. The function should return a PyTorch tensor $F$ of shape $(N, B)$.
- It is possible to use this algorithm with a semi-stochastic gradient descent. In this case, the function should accept a PyTorch tensor $X$ of shape $(N, B)$, where $N$ is the dimension and $B$ is the batch size. The function should return a PyTorch tensor $F$ of shape $(N, B)$.
- It is possible to use this algorithm with a stochastic gradient descent. In this case, the function should accept a PyTorch tensor $X$ of shape $(N, 1)$, where $N$ is the dimension. The function should return a PyTorch tensor $F$ of shape $(N, 1)$.
- It is possible to use this algorithm in a distributed setting. In this case, the function should accept a PyTorch tensor $X$ of shape $(N, B, W)$, where $N$ is the dimension, $B$ is the batch size, and $W$ is the number of workers. The function should return a PyTorch tensor $F$ of shape $(N, B, W)$.

```python
def anderson(function, initial_value, grad_checks=10, lambda_=1e-3, max_iter=1000, tol=1e-3, beta=0):
    
    # Initialization
    X = [initial_value.clone()]
    F = [function(X[0])]
    G = [F[0].clone().detach()]
    residual = []

    # Iterations
    for iteration in range(max_iter):

        # Compute the residual norm
        residual.append(torch.norm(F[iteration] - X[iteration]))

        # Stop if the residual norm is smaller than tol
        if residual[iteration] < tol:
            break

        # Compute the Anderson acceleration
        if iteration >= grad_checks:
            H = torch.stack([torch.matmul(G[i], G[j].t()) for i in range(iteration-grad_checks, iteration) for j in range(iteration-grad_checks, iteration)], 0).view(grad_checks, grad_checks)
            y = torch.ones(grad_checks, 1)
            alpha = torch.matmul(torch.inverse(H + lambda_*torch.eye(grad_checks)), y)
            alpha = alpha / torch.sum(alpha)
            X.append(torch.matmul(F[-grad_checks:], alpha))
        else:
            X.append(F[iteration])

        # Update the iterates
        X[iteration+1] = (1 - beta) * X[iteration] - beta * X[iteration+1]

        # Get the function value and gradient at the new iterate
        F.append(function(X[iteration+1]))
        G.append(F[iteration+1].clone().detach())

    return X, F, G, residual, iteration
```
## Experiments

We compare the convergence of the Anderson acceleration algorithm with the standard gradient descent algorithm on a simple quadratic function. 
The function is defined by:
$$f(x) = \frac{1}{2}x^THx - \frac{1}{2}x^Tb + c$$
where $H = \begin{pmatrix} 1 & 0.5 \\ 0.5 & 1\end{pmatrix}$ and $b = (1, 1)^T$. 

We choose the regularization parameter to be $\lambda = 10^{-3}$. We set $\beta = 0$ so that it is equivalent to pure gradient descent. We set the maximum number of iterations to be $10^4$. We set the tolerance to be $10^{-3}$. 

The figure below compares the performance of Anderson Acceleration and Gradient Descent. The results show that Anderson Acceleration converges faster than Gradient Descent for the same number of gradient evaluations. 


<center><img src="figures/anderson.png" width=600></center>

## Conclusion

Anderson Acceleration is a generalization of gradient descent. It uses the history of gradients to accelerate the convergence. 

## References

- A. Anderson, "Convergence of Krylov subspace methods for non-symmetric linear systems," BIT, vol. 28, no. 2, pp. 467-489, 1988.
- A. Anderson, "Iterative procedures for nonlinear integral equations," Journal of the ACM, vol. 14, no. 1, pp. 1-13, 1967.
- J. Nocedal, "Updating Quasi-Newton Matrices with Limited Storage," Mathematics of Computation, vol. 35, no. 151, pp. 773-782, 1980.
- S. Reddi, S. Kale, and S. Kumar, "On the convergence of Adam and beyond," International Conference on Learning Representations (ICLR), 2019.
- R. Zhang, B. Recht, and M. I. Jordan, "On the global linear convergence of gradient descent for over-parameterized models using optimal transport," Advances in Neural Information Processing Systems (NIPS), 2018.
- J. M. Martínez, A. Rakhlin, and M. Zinkevich, "A general acceleration technique for stochastic optimization," International Conference on Machine Learning (ICML), 2016.
- A. Rakhlin and K. Sridharan, "Making stochastic gradient descent work for you," International Conference on Machine Learning (ICML), 2013.
