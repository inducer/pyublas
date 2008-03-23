from pyublas._internal import dblmat, dblvec
import numpy
print dblvec(numpy.arange(17, dtype=numpy.cdouble))
a = numpy.array([[1,2],[3,4]])
print a.dtype
print a
print dblmat(a)

from pyublas import sparse
u =  sparse({2:{3:17}}, shape=(6,6))
v =  sparse({5:{2:17}}, shape=(6,6))
print u+v
