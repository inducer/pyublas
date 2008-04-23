.. highlight:: c++

The C++ Vector and Matrix Types
===============================

PyUblas defines Ublas-compatible C++ types, namelyvector and matrix types,
:ctype:`numpy_vector` and :ctype:`numpy_matrix`. These are defined in
:file:`pyublas/numpy.hpp`.

Introduction
------------

:ctype:`numpy_vector` is derived from Ublas's ``vector`` and may be used in the
same places as the latter, in the same ways.  There is only one difference:

  :ctype:`numpy_vector` will not copy your data unless you ask it to.

:ctype:`numpy_vector` internally uses Numpy arrays to store data.  Unlike the
native Ublas ``vector<T>``, :ctype:`numpy_vector` has reference or "handle"
semantics: Copy-constructing and assigning only makes a new reference to the
existing vector's data. Unlike native Ublas, it will not make a copy. Consider
this example::

  numpy_vector<double> a(5);
  std::fill(a.begin(), a.end(), 17);
    
  numpy_vector<double> b(a); // only references a's data
  b[2] = 0;

  std::cout << a[2] << std::endl; // prints "0".

If you *want* the data to be copied, you must say so explicitly::

  numpy_vector<double> c(a.size());
  c.assign(a); // actually copies a's data
  c[2] = 17;

  std::cout << a[2] << std::endl; // still prints "0".
  std::cout << c[2] << std::endl; // prints "17".

If you follow these simple rules, it's easy to write code that works the same
for both :ctype:`ublas::vector` and :ctype:`numpy_vector`. (Note that here, as
in the rest of this documentation, the same discussion applies without change
to :ctype:`numpy_matrix` as well.) 

**CAUTION:** The `assign` method does not resize its target to the size of its operand--this is something you have to do by hand if you use the above recipe.

You should be aware of another, small difference: Because :ctype:`numpy_vector` keeps
its data reference inside a Numpy array object, indexed access is a good deal
slower than iterator access. Iterators achieve the same speed as "regular"
Ublas, while indexed access adds an extra level of pointer lookup. As is true
of much of the rest of C++: *Use iterators whenever possible.*

Reference
---------

``#include <pyublas/numpy.hpp``

.. ctype:: numpy_array
  
    ``template <class ValueType>``, in namespace ``pyublas``.

    Only members that are not already part of the 
    `Boost.Ublas "Storage" Concept <http://www.boost.org/doc/libs/1_35_0/libs/numeric/ublas/doc/storage_concept.htm>`_
    are shown.

    Public type definitions::

      typedef std::size_t size_type;
      typedef std::ptrdiff_t difference_type;
      typedef ValueType value_type;
      typedef const ValueType &const_reference;
      typedef ValueType &reference;
      typedef const ValueType *const_pointer;
      typedef ValueType *pointer;

    .. cfunction:: numpy_array numpy_array::constructor()
                   numpy_array numpy_array::constructor(size_type n)
                   numpy_array numpy_array::constructor(size_type n, const value_type &v)
                   numpy_array numpy_array::constructor(int ndim, const npy_intp *dims)
                   numpy_array numpy_array::constructor(const boost::python::handle<> &obj)

    .. cfunction:: size_type numpy_array::ndim()

        A ``const`` member function.
    .. cfunction:: const npy_intp *numpy_array::dims()

        A ``const`` member function.

    .. cfunction:: const npy_intp *numpy_array::strides()

        A ``const`` member function.

    .. cfunction:: npy_intp numpy_array::min_stride()

        A ``const`` member function.

    .. cfunction:: npy_intp numpy_array::itemsize()

        A ``const`` member function.

    .. cfunction:: bool numpy_array::writable()

        A ``const`` member function.

    .. cfunction:: void numpy_array::reshape(int ndim, const npy_intp *dims, NPY_ORDER order=NPY_CORDER)

    .. cfunction:: value_type *numpy_array::data()

    .. cfunction:: const value_type *numpy_array::data()

        A ``const`` member function.

    .. cfunction:: const boost::python::handle<> &numpy_array::handle()

        A ``const`` member function.

    .. cfunction:: boost::python::handle<> &numpy_array::handle() 

.. ctype:: numpy_vector

    ``template <class ValueType>``, in namespace ``pyublas``.

.. ctype:: numpy_matrix

    ``template <class ValueType, class Orientation=boost::numeric::ublas::row_major>``, 
    in namespace ``pyublas``.

Interacting with Boost.Bindings
-------------------------------

PyUblas contains special code to support interacting with the `Boost.Bindings
<http://mathema.tician.de/software/boost-bindings>`_ library.

If you want to activate this support, define the macro 
:cmacro:`PYUBLAS_HAVE_BOOST_BINDINGS` before include :file:`pyublas/numpy.hpp`.

Boost.Bindings works seamlessly with :ctype:`numpy_vector`. For 
:ctype:`numpy_matrix`, you need to explicitly downcast it to the
:ctype:`ublas::matrix` type. You may do so by simply calling the
``.as_ublas()`` method.
