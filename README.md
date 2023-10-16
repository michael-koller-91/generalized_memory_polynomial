# Generalized Memory Polynomials

Two kinds of memory polynomials `f` are implemented:
- without cross-terms (`cross_terms = False`)
```math
    x_{\text{out}}(n) = f(x_\text{in}[n]) = \sum_{m=0}^{M-1} \sum_{k=0}^{K-1} a_{mk} x_{\text{in}}(n - m) \cdot | x_{\text{in}}(n - m) |^k
```

- with cross-terms (`cross_terms = True`)
```math
    x_{\text{out}}(n) = f(x_\text{in}[n]) = \sum_{m=0}^{M-1} c_m x_{\text{in}}(n - m)
        + \sum_{m=0}^{M-1} \sum_{j=0}^{M-1} \sum_{k=1}^{K-1} a_{mjk} x_{\text{in}}(n - m) \cdot | x_{\text{in}}(n - m) |^k
```

## Example

