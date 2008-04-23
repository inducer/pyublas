.. highlight:: c++

Automatic From-Python Conversion
================================

Shapes, Ranks and Sizes
-----------------------

Two simple rules:

* Any shape of array can be converted to a :ctype:`numpy_vector`.

* Only 2D arrays of the right element order (i.e. column-/row-major) can be
  converted to :ctype:`numpy_matrix`.

References and Pointers
-----------------------

PyUblas registers from-Python converters only for *rvalues*. This means that
PyUblas will not successfully wrap functions that expect a
:ctype:`numpy_vector` lvalue, i.e. a reference or a pointer::

  void do_stuff1(numpy_vector<double> &v) { ... }       // (1) WILL NOT WORK
  void do_stuff2(numpy_vector<double> *v) { ... }       // (2) WILL NOT WORK
  void do_stuff3(const numpy_vector<double> &v) { ... } // (3) OK, can't modify v
  void do_stuff4(numpy_vector<double> v) { ... }        // (4) OK, can modify v

Note that versions (1) and (2) will successfully compile, but any invocation to
``do_stuff1`` and ``do_stuff2`` from Python will simply fail, with Boost.Python
reporting that no valid overload could be found. Version (3) is fine, because
const references count as rvalues. Version (4) is the recommended version if
you want to modify `v` in-place. Recall that :ctype:`numpy_vector` is a handle class
for an existing, separate chunk of numpy data.

For a similar reason, you may only use Boost.Python's ``extract`` facility on
non-references and non-pointers of :ctype:`numpy_vector` type::

  boost::python::object o;
  
  // (1) WRONG, will throw exception
  numpy_vector<double> &u1 = boost::python::extract<numpy_vector<double> &>(o);
  
  // (2) WRONG, will throw exception
  const numpy_vector<double> &u1 = boost::python::extract<const numpy_vector<double> &>(o);
  
  // (3) OK
  numpy_vector<double> u1 = boost::python::extract<numpy_vector<double> >(o);

Since there is no underlying :ctype:`numpy_vector` instance, no reference can be
extracted that would outlive the `extract<>()` constructor call. Therefore (1)
and (2) are invalid. Because of the reference semantics, (3) is probably what
you want anyway.

Note that :ctype:`numpy_vector` also has a constructor accepting a
``boost::python::handle<>`` (which is a thin wrapper around a ``PyObject *``).
This may also be used to conveniently construct a :ctype:`numpy_vector` from a
known Python instance.

When conversion fails
---------------------

Even if you heed all the advice in the previous section, sometimes Boost.Python
still complains that no valid overload can be found. Common reasons include:

* You expect to be handling a float array, but you actually got an int (or
  different dtype) array. To be consistent, PyUblas will not copy-and-cast in
  this situation. If necessary, use :func:`numpy.asarray` in your Python code.
  (I'm debating whether to add C++ syntax to say "copying is ok here". If you
  have input, let me know.)

* If converting a matrix, the row/column-major setting may not match what Ublas
  is expecting. By default, :ctype:`numpy_matrix` is row-major, as is :class:`numpy.array`
  when creating a 2D array.

The function :func:`pyublas.why_not` can help you debug these cases.

