The :mod:`pyublas` Python module
================================

Sparse Matrix wrappers
----------------------

**CAUTION:** In PyUblas version 0.91 and later, the sparse wrappers are an
optional feature that has to be enabled at `configure` time with the option
`--with-sparse-wrappers`. You may check for their presence at runtime by
looking at the result of `pyublas.has_sparse_wrappers()`.

In addition to `numpy_vector` and `numpy_matrix`, PyUblas also wraps Ublas's
powerful sparse matrix machinery into Python objects. The interface to these
functions is somewhat like `numpy`'s own and is found in the `pyublas` name
space.

Here's a brief demo::

  import numpy
  import pyublas

  a = pyublas.zeros((5,5), flavor=pyublas.SparseBuildMatrix, dtype=float)

  a[4,2] = 19

  b = numpy.random.randn(2,2)
  a.add_block(2, 2, b)

  a_fast = pyublas.asarray(a, flavor=pyublas.SparseExecuteMatrix)

  vec = numpy.random.randn(5)

  res = a_fast * vec

  print a_fast
  print res

This prints something like::

  sparse({2: {2: 0.774217588463, 3: -1.5320702452},
          3: {2: 0.118048365647, 3: 1.05028340411},
          4: {2: 19.0}},
         shape=(5, 5), flavor=SparseExecuteMatrix)
  [ 0.          0.         -0.60793048  0.13384055 -8.28513612]

The `SparseBuildMatrix` flavor is designed for fastest possible assembly of
sparse matrices, while the `SparseExecuteMatrix` flavor is made for the fastest
possible matrix-vector product. There's much more functionality here--don't be
afraid to peek into the source code.

