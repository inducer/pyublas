.. highlight:: c++

The C++ Vector and Matrix Types
===============================

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

Interaction with Boost.Bindings
-------------------------------

If you use the `Boost.Bindings
<http://mathema.tician.de/software/boost-bindings>`_ library to access existing
Fortran or C code, your code will continue to work for :ctype:`numpy_vector`
unmodified. For :ctype:`numpy_matrix`, there is a slight catch that may be
fixed in a future version. For the :ctype:`matrix_traits` to properly recognize
:ctype:`numpy_matrix`, you need to explicitly downcast it to the
:ctype:`ublas::matrix` type. You may do so by simply calling the
``.as_ublas()`` method on :ctype:`numpy_matrix`.

