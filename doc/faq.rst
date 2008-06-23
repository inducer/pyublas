Frequently Asked Questions
==========================

Where do the headers get installed?
-----------------------------------

Its easiest to find this location using this Python code snippet::

    from imp import find_module
    file, pathname, descr = find_module("pyublas")
    from os.path import join
    return join(pathname, "..", "include")

Since PyUblas requires numpy, you'll need a similar snippet for that, too::

    from imp import find_module
    file, pathname, descr = find_module("numpy")
    from os.path import join
    return join(pathname, "core", "include")

What about 0-dimensional arrays?
--------------------------------

0-dimensional arrays are supported by PyUblas, they can be converted
to :ctype:`numpy_vector` instances of length 1.

.. _faq-overload-failure:

My wrapped function is not found by Boost.Python. Help!
-------------------------------------------------------

Even if you heed all the advice in :ref:`frompython`, Boost.Python
will sometimes still complain that no valid overload can be found.
Common reasons include:

* You expect to be handling a float array, but you actually got an int (or
  different dtype) array. To be consistent, PyUblas will not copy-and-cast in
  this situation. If necessary, use :func:`numpy.asarray` in your Python code.
  (I'm debating whether to add C++ syntax to say "copying is ok here". If you
  have input, let me know.)

* If converting a matrix, the row/column-major setting may not match what Ublas
  is expecting. By default, :ctype:`numpy_matrix` is row-major, as is :class:`numpy.array`
  when creating a 2D array.

The function :func:`pyublas.why_not` can help you debug these cases.

User-visible Changes
====================

PyUblas 0.93
------------

* Negative strides are supported. Slice handling was cleaned up and should be 
  correct now.

* :class:`invalid_ok` was added.

* :class:`numpy_strided_vector` was added as another way of transparently dealing
  with non-contiguous slices.

* :cfunc:`numpy_vector::min_stride` is gone. It was ill-specified and not capable
  of doing what it promised to do.

* :mod:`numpy` forces every C/C++ module that uses its functionality to call
  :cfunc:`import_array`. PyUblas has a clever mechanism that does this for you.
  This mechanism was not correct previously, it would often fail when a 
  particular piece of code was not inlined.
