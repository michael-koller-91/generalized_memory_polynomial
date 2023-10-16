# Generalized Memory Polynomials

Two kinds of memory polynomials `f` are implemented:
- without cross-terms (`cross_terms = False`)
```math
    x_{\text{out}}(n) = f(x_\text{in}(n)) = \sum_{m=0}^{M-1} \sum_{k=0}^{K-1} a_{mk} \cdot x_{\text{in}}(n - m) \cdot | x_{\text{in}}(n - m) |^k
```

- with cross-terms (`cross_terms = True`)
```math
    x_{\text{out}}(n) = f(x_\text{in}(n)) = \sum_{m=0}^{M-1} c_m \cdot x_{\text{in}}(n - m)
        + \sum_{m=0}^{M-1} \sum_{j=0}^{M-1} \sum_{k=1}^{K-1} a_{mjk} \cdot x_{\text{in}}(n - m) \cdot | x_{\text{in}}(n - j) |^k
```

The parameter $M$ is called memory depth and the parameter $K$ is called degree.

To determine the coefficients ($a_{mk}$ or $c_m$ and $a_{mjk}$),
a data matrix $X$ is constructed which collects all terms of the form $x_{\text{in}}(n - m)$ and $|x_{\text{in}}(n - m)|^k$.
Then, if also the samples $x_{\text{out}}(n)$ are collected in a vector $x_{\text{out}}$ and if all coefficients are collected in a vector $c$,
the solution is determined as
```math
    c = (X^{\mathrm{H}} X + \alpha I)^{-1} X^{\mathrm{H}} x_{\text{out}}
```
with a regularization parameter $\alpha \geq 0$ (`alpha`).

## Example

To instantiate a memory polynomial without cross-terms and with $M=3$, $K=2$, $\alpha = 0.1$, write
```python
mem_pol = gmp.MemoryPolynomial(degree=2, mem_depth=3, alpha=0.1, cross_terms=False)
```
and determine the coefficients via
```python
c = mem_pol.fit(x_in, x_out)
```
To predict/equalize/postdistort some data `y_in`, use
```python
y_out = mem_pol(y_in)
```

## Reference

Morgan, Ma, Kim, Zierdt, Pastalan, "A Generalized Memory Polynomial Model for Digital Predistortion of Power Amplifiers," IEEE Trans. Signal Process., 2006.
