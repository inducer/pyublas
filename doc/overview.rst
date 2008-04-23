
PyUblas consists of two components:

* Ublas-compatible C++ vector and matrix types, ``pyublas::numpy_vector<T>`` and
  ``pyublas::numpy_matrix<T>``. These are defined in :file:`pyublas/numpy.hpp`.

* Automatic from/to-Python converters for Boost.Python that allow Numpy arrays
  as Python arguments to exposed C++ functions. These are available as soon as
  you ``import pyublas`` in your Python code.


