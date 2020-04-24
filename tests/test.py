
import cga_cpp  

from clifford.g3c import *
import numpy as np
import time
import numba
imt_func = layout.imt_func
gmt_func = layout.gmt_func


# Set up a super basic function
def basic_op(a, b):
    return ((a|b)*b)**2

# Set up a super basic function
@numba.njit
def basic_op_jitted(aval, bval):
    res = gmt_func(imt_func(aval, bval), bval)
    return gmt_func(res, res)


# Visually check correctness
a = layout.randomMV()(1,2,3,4,5) 
b = layout.randomMV()(1,2,3,4,5)
print('Python')
print(basic_op(a, b))
print()
print('Python jitted')
print(layout.MultiVector(basic_op_jitted(a.value, b.value)))
print()
print('C++')
print(layout.MultiVector(cga_cpp.basic_op(a.value, b.value)))
print()

# Check 1000 random test cases
for i in range(1000):
    a = layout.randomMV()(1,2,3,4,5)
    b = layout.randomMV()(1,2,3,4,5)
    res = cga_cpp.basic_op(a.value, b.value)
    np.testing.assert_allclose(basic_op(a, b).value, res, rtol=1E-6, atol=1E-8)
    np.testing.assert_allclose(basic_op_jitted(a.value, b.value), res, rtol=1E-6, atol=1E-8)

# Time and compare
start_time = time.time()
for i in range(10000):
    basic_op(a, b)
end_time = time.time()
print('Python')
print(end_time - start_time)
print()

# Time and compare
start_time = time.time()
for i in range(10000):
    basic_op_jitted(a.value, b.value)
end_time = time.time()
print('Python jitted')
print(end_time - start_time)
print()

start_time = time.time()
for i in range(10000):
    cga_cpp.basic_op(a.value, b.value)
end_time = time.time()
print('C++')
print(end_time - start_time)

