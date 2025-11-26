from sympy import symbols, cos, fourier_series, pi

x = symbols('x')
f = cos(x)**8

# Compute Fourier series over the interval [-pi, pi]
fs = fourier_series(f, (x, -pi, pi))

print(fs.truncate(n=10))