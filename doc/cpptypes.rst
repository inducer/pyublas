.. highlight:: c++

The C++ Vector and Matrix Types
===============================

PyUblas defines Ublas-compatible C++ types, namely
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

Reference Documentation
-----------------------

``#include <pyublas/numpy.hpp>``

Note that due to the limitations of this document format, 
a C++ member function that should normally be called 
``numpy_vector::as_strided`` is here written as 
``numpy_vector_as_strided``.

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

    .. cfunction:: constructor numpy_array()
                   explicit_constructor numpy_array(size_type n)
                   constructor numpy_array(size_type n, const value_type &v)
                   constructor numpy_array(int ndim, const npy_intp *dims)
                   constructor numpy_array(const boost::python::handle<> &obj)

    .. cfunction:: size_type numpy_array_ndim()

        A ``const`` member function.
        
    .. cfunction:: const npy_intp *numpy_array_dims()

        A ``const`` member function.

    .. cfunction:: const npy_intp *numpy_array_strides()

        A ``const`` member function.

    .. cfunction:: npy_intp numpy_array_min_stride()

        A ``const`` member function.

    .. cfunction:: npy_intp numpy_array_itemsize()

        A ``const`` member function.

    .. cfunction:: bool numpy_array_writable()

        A ``const`` member function.

    .. cfunction:: void numpy_array_reshape(int ndim, const npy_intp *dims, NPY_ORDER order=NPY_CORDER)

    .. cfunction:: value_type *numpy_array_data()

    .. cfunction:: const value_type *numpy_array_data()

        A ``const`` member function.

    .. cfunction:: const boost::python::handle<> &numpy_array_handle()

        A ``const`` member function.

    .. cfunction:: boost::python::handle<> &numpy_array_handle() 

.. ctype:: numpy_vector

    ``template <class ValueType>``, in namespace ``pyublas``.

    .. cfunction:: constructor numpy_vector_constructor()
                   constructor numpy_vector(const numpy_array<ValueType> &s)
                   constructor numpy_vector(int ndim, const npy_intp *dims)
                   explicit_constructor numpy_vector(typename super::size_type size)
                   constructor numpy_vector(size_type size, const value_type &init)
                   constructor numpy_vector(const numpy_vector &v)
                   constructor numpy_vector(const boost::numeric::ublas::vector_expression<AE> &ae)

        Construct a new :ctype:`numpy_vector` instance.

        The ``(ndim, dims)`` constructor form can be used to specify
        the Python-side shape of the array at construction time.

        Observe that PyObject handles are implicitly convertible
        to :ctype:`numpy_array`, so that you can invoke the 
        constructor simply by feeding it a ``boost::python::handle``.

    .. cfunction:: size_type numpy_vector_ndim()

        Return the number of dimensions of this array.

        A ``const`` member function.
        
    .. cfunction:: const npy_intp *numpy_vector_dims()

        Return an array of :cfunc:`numpy_vector_ndim` entries,
        each of which is the size of the array along one dimension. 
        in *elements*. 

        A ``const`` member function.

    .. cfunction:: const npy_intp *numpy_vector_strides()

        Return an array of :cfunc:`numpy_vector_ndim` entries,
        each of which is the stride along one dimension, in 
        *bytes*. Divide by :cfunc:`numpy_vector_itemsize` 
        to convert this to element-wise strides.

        A ``const`` member function.

    .. cfunction:: npy_intp numpy_vector_min_stride()

        The smallest stride used in the underlying array, in bytes.
        Divide by :cfunc:`numpy_vector_itemsize` to convert this to
        element-wise strides.

        A ``const`` member function.

    .. cfunction:: npy_intp numpy_vector_itemsize()
        
        Return the size (in bytes) of each element of the array.

        A ``const`` member function.
    .. cfunction:: bool numpy_vector_writable()

        A ``const`` member function.

    .. cfunction:: ValueType &numpy_vector_sub(npy_intp i) 
                   ValueType &numpy_vector_sub(npy_intp i, npy_intp j) 
                   ValueType &numpy_vector_sub(npy_intp i, npy_intp j, npy_intp k) 
                   ValueType &numpy_vector_sub(npy_intp i, npy_intp j, npy_intp k, npy_intp l) 

        Return the element at the index (i), (i,j), (i,j,k),
        (i,j,k,l). It is up to you to ensure that the array
        has the same number of dimensions, otherwise the results
        are undefined.

        Also available as ``const`` member functions.

    .. cfunction:: void numpy_vector_reshape(int ndim, const npy_intp *dims, NPY_ORDER order=NPY_CORDER)
        
        Same operation as :func:`numpy.reshape`.

    .. cfunction:: boost::numeric::ublas::vector_slice<numpy_vector> numpy_vector_as_strided()
        
        Return a view of the array that seems contiguous, by 
        only looking at every :cfunc:`numpy_vector_min_stride`'th 
        element.

        Also available as a ``const`` member function.

    .. cfunction:: boost::vector<ValueType> &numpy_vector_as_ublas() 

        Downcast this instance to the underlying 
        ``boost::numeric::ublas::vector<ValueType>``.

        Also available as a ``const`` member function.

    .. cfunction:: boost::python::handle<> numpy_vector_to_python()

        Return a Boost.Python ``handle`` (which is essentially an
        auto-refcounting ``PyObject *``) to the underlying Numpy
        array.

        A ``const`` member function.

.. ctype:: numpy_matrix

    ``template <class ValueType, class Orientation=boost::numeric::ublas::row_major>``, 
    in namespace ``pyublas``.

    .. cfunction:: numpy_matrix()
                   numpy_matrix(size_type size1, size_type size2)
                   numpy_matrix(size_type size1, size_type size2, const value_type &init)
                   numpy_matrix(size_type size1, size_type size2, const array_type &data)
                   numpy_matrix(const typename super::array_type &data)
                   numpy_matrix_constructor(const numpy_matrix &m)
                   numpy_matrix_constructor(const boost::numeric::ublas::matrix_expression<AE> &ae)

        Observe that PyObject handles are implicitly convertible
        to :ctype:`numpy_array`, so that you can invoke the 
        constructor simply by feeding it a ``boost::python::handle``.

    .. cfunction:: boost::matrix<ValueType, Orientation> &numpy_matrix_as_ublas() 

        Also available as a ``const`` member function.

    .. cfunction:: boost::python::handle<> numpy_matrix_to_python()

        A ``const`` member function.

Interacting with Boost.Bindings
-------------------------------

PyUblas contains special code to support interacting with the `Boost.Bindings
<http://mathema.tician.de/software/boost-bindings>`_ library.

If you want to activate this support, define the macro 
:cmacro:`PYUBLAS_HAVE_BOOST_BINDINGS` before including :file:`pyublas/numpy.hpp`.

Boost.Bindings works seamlessly with :ctype:`numpy_vector`. For 
:ctype:`numpy_matrix`, you need to explicitly downcast it to the
:ctype:`ublas::matrix` type. You may do so by simply calling the
:cfunc:`as_ublas` method.
