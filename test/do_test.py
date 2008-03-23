from pyublas._internal import dblmat, dblvec
import numpy
print dblvec(numpy.arange(17, dtype=numpy.cdouble))
a = numpy.array([[1,2],[3,4]])
print a.dtype
print a
print dblmat(a)
