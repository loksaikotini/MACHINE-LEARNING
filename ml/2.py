
import math

print("1. Factorial of 6:", math.factorial(6))
print("2. Square root of 225:", math.sqrt(225))
print("3. Natural log of 100:", math.log(100))
print("4. e raised to 2 (e^2):", math.exp(2))
print("5. 5 raised to power 2:", math.pow(5, 2))
print("6. Sine of 60 degrees:", math.sin(math.radians(60))) 
print("7. GCD of 36 and 60:", math.gcd(36, 60))


import numpy as np

arr1 = np.array([4, 2, 7])
arr2 = np.array([[2, 6], [3, 1]])

print("8. 1D array:", arr1)
print("9. 2D array:\n", arr2)
print("10. Mean of arr1:", np.mean(arr1))

print("11. Dot product of [1, 2] and [3, 4]:", np.dot([1, 2], [3, 4]))
print("12. Inverse of [[1, 2], [3, 4]]:\n", np.linalg.inv(arr2))


import statistics as stats

data = [10, 20, 30, 40, 50]

print("13. Mean of data:", stats.mean(data))
print("14. Median of data:", stats.median(data))
print("15. Mode of data:", stats.mode([1, 2, 8, 6, 6, 6, 4]))
print("16. Variance of data:", stats.variance(data))

from scipy.optimize import minimize
from scipy.integrate import quad, dblquad

def f(x): return x[0]**2 + x[1]**2
result = minimize(f, [1, 1])
print("18. Minimum of x^2 + y^2 is at:", result.x)

area = quad(lambda x: x**2, 0, 2)
print("19. Integral of x^2 from 0 to 2:", area[0])

# 20. Double integral of x*y from x=0..1, y=0..2
double_area = dblquad(lambda x, y: x * y, 0, 1, lambda x: 0, lambda x: 2)
print("20. Double integral of x*y over x=0..1 and y=0..2:", double_area[0])
