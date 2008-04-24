//
// Copyright (c) 2008 Andreas Kloeckner
//
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and
// that both that copyright notice and this permission notice appear
// in supporting documentation.  The authors make no representations
// about the suitability of this software for any purpose.
// It is provided "as is" without express or implied warranty.
//




#ifndef _AFAYFYDASDFAH_PYUBLAS_HEADER_SEEN_NUMPY_HPP
#define _AFAYFYDASDFAH_PYUBLAS_HEADER_SEEN_NUMPY_HPP




#include <numeric>
#include <complex>
#include <pyublas/python_helpers.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/python.hpp>
#include <boost/foreach.hpp>
#include <numpy/arrayobject.h>





namespace pyublas
{
  static struct array_importer
  {
    array_importer()
    { import_array(); }
  } _array_importer;

  inline NPY_TYPES get_typenum(bool) { return NPY_BOOL; }
  inline NPY_TYPES get_typenum(npy_bool) { return NPY_BOOL; }
  inline NPY_TYPES get_typenum(npy_byte) { return NPY_BYTE; }
  // NPY_TYPES get_typenum(npy_ubyte) { return NPY_UBYTE; }
  inline NPY_TYPES get_typenum(npy_short) { return NPY_SHORT; }
  inline NPY_TYPES get_typenum(npy_ushort) { return NPY_USHORT; }
  inline NPY_TYPES get_typenum(npy_int) { return NPY_INT; }
  inline NPY_TYPES get_typenum(npy_uint) { return NPY_UINT; }
  inline NPY_TYPES get_typenum(npy_long) { return NPY_LONG; }
  inline NPY_TYPES get_typenum(npy_ulong) { return NPY_ULONG; }
  inline NPY_TYPES get_typenum(npy_longlong) { return NPY_LONGLONG; }
  inline NPY_TYPES get_typenum(npy_ulonglong) { return NPY_ULONGLONG; }
  inline NPY_TYPES get_typenum(npy_float) { return NPY_FLOAT; }
  inline NPY_TYPES get_typenum(npy_double) { return NPY_DOUBLE; }
  inline NPY_TYPES get_typenum(npy_longdouble) { return NPY_LONGDOUBLE; }
  inline NPY_TYPES get_typenum(npy_cfloat) { return NPY_CFLOAT; }
  inline NPY_TYPES get_typenum(npy_cdouble) { return NPY_CDOUBLE; }
  inline NPY_TYPES get_typenum(npy_clongdouble) { return NPY_CLONGDOUBLE; }
  inline NPY_TYPES get_typenum(std::complex<float>) { return NPY_CFLOAT; }
  inline NPY_TYPES get_typenum(std::complex<double>) { return NPY_CDOUBLE; }
  inline NPY_TYPES get_typenum(std::complex<long double>) { return NPY_CLONGDOUBLE; }
  inline NPY_TYPES get_typenum(boost::python::object) { return NPY_OBJECT; }
  inline NPY_TYPES get_typenum(boost::python::handle<>) { return NPY_OBJECT; }
  /* NPY_STRING, NPY_UNICODE unsupported for now */




  // tool functions -----------------------------------------------------------
  inline
  npy_intp size_from_dims(int ndim, const npy_intp *dims)
  {
    if (ndim != 0)
      return std::accumulate(dims, dims+ndim, 1, std::multiplies<npy_intp>());
    else
      return 1;
  }




  // ublas storage array ------------------------------------------------------
  template <class T>
  class numpy_array
  {
    private:
      // Life support for the numpy array.
      boost::python::handle<>         m_numpy_array;

    public:
      typedef std::size_t size_type;
      typedef std::ptrdiff_t difference_type;
      typedef T value_type;
      typedef const T &const_reference;
      typedef T &reference;
      typedef const T *const_pointer;
      typedef T *pointer;

      // Construction and destruction
      numpy_array()
      {
        m_numpy_array = boost::python::handle<>(
            PyArray_SimpleNew(0, 0, get_typenum(T())));
      }

      numpy_array(size_type n)
      {
        npy_intp dims[] = { n };
        m_numpy_array = boost::python::handle<>(
            PyArray_SimpleNew(1, dims, get_typenum(T())));
      }

      numpy_array(int ndim_, const npy_intp *dims_)
      {
        m_numpy_array = boost::python::handle<>(
            PyArray_SimpleNew(
              ndim_, 
              const_cast<npy_intp *>(dims_), 
              get_typenum(T())));
      }

      numpy_array(size_type n, const value_type &v)
      {
        if (size)
        {
          npy_intp dims[] = { n };
          m_numpy_array = boost::python::handle<>(
              PyArray_SimpleNew(1, dims, get_typenum(T())));
          fill(begin(), end(), v);
        }
      }

      /* MISSING
      Range constructor 	X(i, j) 	
      i and j are Input Iterators whose value type is convertible to T 	X
      */

      numpy_array(const boost::python::handle<> &obj)
        : m_numpy_array(obj)
      {
        if (!PyArray_Check(obj.get()))
          PYUBLAS_PYERROR(TypeError, "argument is not a numpy array");
        if (PyArray_TYPE(obj.get()) != get_typenum(T()))
          PYUBLAS_PYERROR(TypeError, "argument is numpy array of wrong type");
        if (!PyArray_CHKFLAGS(obj.get(), NPY_ALIGNED))
            PYUBLAS_PYERROR(ValueError, "argument array is not aligned");
        if (PyArray_CHKFLAGS(obj.get(), NPY_NOTSWAPPED))
            PYUBLAS_PYERROR(ValueError, "argument array does not have native endianness");
        if (PyArray_ITEMSIZE(obj.get()) != sizeof(T))
            PYUBLAS_PYERROR(ValueError, "itemsize does not match size of target type");
      }

    private:
      void resize_internal (size_type new_size, value_type init, bool preserve = true) 
      {
        size_type old_size;
        if (m_numpy_array.get())
          old_size = size();
        else
        {
          preserve = false;
          old_size = 0;
        }

        if (new_size != old_size) 
        {
          npy_intp dims[] = { new_size };
          boost::python::handle<> new_array = boost::python::handle<>(
              PyArray_SimpleNew(1, dims, get_typenum(T())));
          pointer new_data = reinterpret_cast<T *>(
              PyArray_DATA(new_array.get()));

          if (preserve) 
          {
            std::copy(data(), data() + std::min(new_size, old_size), new_data);
            std::fill(new_data + std::min(new_size, old_size), new_data + new_size, init);
          }

          m_numpy_array = new_array;
        }
      }

    public:
      void resize (size_type size) {
        resize_internal (size, value_type (), false);
      }
      void resize (size_type size, value_type init) {
        resize_internal (size, init, true);
      }

      size_type size() const 
      { 
        if (ndim() != 0)
          return min_stride()/sizeof(T)*size_from_dims(ndim(), dims());
        else
          return 1;
      }

      // metadata
      size_type ndim() const 
      { return PyArray_NDIM(m_numpy_array.get()); }
      const npy_intp *dims() const 
      { return PyArray_DIMS(m_numpy_array.get()); }
      const npy_intp *strides() const 
      { return PyArray_STRIDES(m_numpy_array.get()); }
      npy_intp min_stride() const
      {
        if (ndim() == 0)
          return itemsize();
        else
          return *std::min_element(strides(), strides()+ndim());
      }
      npy_intp itemsize() const
      { return sizeof(T); }
      bool writable() const
      { return PyArray_ISWRITEABLE(m_numpy_array.get()); }

      // shape manipulation
      void reshape(int ndim_, const npy_intp *dims_, 
          NPY_ORDER order=NPY_CORDER)
      {
        PyArray_Dims d = { const_cast<npy_intp *>(dims_), ndim_ };
        m_numpy_array = boost::python::handle<>(
            PyArray_Newshape(
              (PyArrayObject *) m_numpy_array.get(), &d, order));
      }

      // Raw data access
      T *data()
      {
        return reinterpret_cast<T *>(
            PyArray_DATA(m_numpy_array.get()));
      }

      const T *data() const
      {
        return reinterpret_cast<const T *>(
            PyArray_DATA(m_numpy_array.get()));
      }

      // Element access
      const_reference operator [] (size_type i) const 
      {
        BOOST_UBLAS_CHECK(i < size(), boost::numeric::ublas::bad_index());
        return data()[i];
      }

      reference operator [] (size_type i) 
      {
        BOOST_UBLAS_CHECK(i < size(), boost::numeric::ublas::bad_index());
        return data()[i];
      }

      // Assignment
      numpy_array &operator=(const numpy_array &a) 
      {
        m_numpy_array = a.m_numpy_array;
        return *this;
      }

      numpy_array &assign_temporary(numpy_array &a) 
      {
        m_numpy_array = a.m_numpy_array;
        return *this;
      }

        // Swapping
      void swap (numpy_array &a) 
      {
        if (this != &a)
          std::swap(m_numpy_array, a.m_numpy_array);
      }

      friend void swap(numpy_array &a1, numpy_array &a2) 
      {
        a1.swap (a2);
      }

      // Iterators simply are pointers.

      typedef const_pointer const_iterator;

      const_iterator begin () const 
      {
        return data();
      }

      const_iterator end () const 
      {
        return data() + size();
      }

      typedef pointer iterator;

      iterator begin () 
      {
        return data();
      }

      iterator end () 
      {
        return data() + size();
      }

      // Reverse iterators
      typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
      typedef std::reverse_iterator<iterator> reverse_iterator;

      const_reverse_iterator rbegin() const 
      { return const_reverse_iterator(end()); }

      const_reverse_iterator rend() const 
      { return const_reverse_iterator(begin ()); }

      reverse_iterator rbegin() 
      { return reverse_iterator(end()); }

      reverse_iterator rend () 
      { return reverse_iterator(begin()); }

      // Data accessor

      const boost::python::handle<> &handle() const
      {
        return m_numpy_array;
      }
      boost::python::handle<> &handle() 
      {
        return m_numpy_array;
      }
  };




  // matrix helper functions --------------------------------------------------
  inline bool is_row_major(boost::numeric::ublas::row_major_tag)
  {
    return true;
  }
  inline bool is_row_major(boost::numeric::ublas::column_major_tag)
  {
    return false;
  }

  template <class OCat, class T>
  typename numpy_array<T>::size_type get_array_size1(numpy_array<T> const &ary)
  {
    typedef numpy_array<T> mat_type;

    if (PyArray_NDIM(ary.handle().get()) != 2)
      throw std::runtime_error("ndarray->matrix converteee has dimension != 2");

    if (PyArray_STRIDE(ary.handle().get(), 1) 
        == PyArray_ITEMSIZE(ary.handle().get()))
    {
      // row-major
      if (!is_row_major(OCat()))
        throw std::runtime_error("input array is not row-major (like the target type)");
      if (!PyArray_CHKFLAGS(ary.handle().get(), NPY_C_CONTIGUOUS))
        throw std::runtime_error("ndarray->matrix converteee is not C-contiguous");
    }
    else if (PyArray_STRIDE(ary.handle().get(), 0) 
        == PyArray_ITEMSIZE(ary.handle().get()))
    {
      // column-major
      if (is_row_major(OCat()))
        throw std::runtime_error("input array is not column-major (like the target type)");
      if (!PyArray_CHKFLAGS(ary.handle().get(), NPY_F_CONTIGUOUS))
        throw std::runtime_error("ndarray->matrix converteee is not F-contiguous");
    }
    else
        throw std::runtime_error("input array is does not have dimension with stride==1");

    return PyArray_DIM(ary.handle().get(), 0);
  }

  template <class T>
  typename numpy_array<T>::size_type get_array_size2(numpy_array<T> const &ary)
  {
    // checking is done in size1()
    return PyArray_DIM(ary.handle().get(), 1);
  }

  template<class T, class L = boost::numeric::ublas::row_major>
  class numpy_matrix;

  template <class T, class L>
  boost::python::handle<> matrix_to_python(numpy_matrix<T, L> const &mat)
  {
    typedef numpy_matrix<T, L> mat_type;
    boost::python::handle<> orig_handle = mat.data().handle();

    npy_intp dims[] = { mat.size1(), mat.size2() };
    boost::python::handle<> result;

    if (is_row_major(typename mat_type::orientation_category()))
    {
      result = boost::python::handle<>(PyArray_New(
          &PyArray_Type, 2, dims, 
          get_typenum(typename mat_type::value_type()), 
          /*strides*/0, 
          PyArray_DATA(orig_handle.get()),
          /* ? */ 0, 
          NPY_CARRAY, NULL));
    }
    else
    {
      result = boost::python::handle<>(PyArray_New(
          &PyArray_Type, 2, dims, 
          get_typenum(typename mat_type::value_type()), 
          /*strides*/0, 
          PyArray_DATA(orig_handle.get()),
          /* ? */ 0, 
          NPY_FARRAY, NULL));
    }

    PyArray_BASE(result.get()) = boost::python::handle<>(orig_handle).release();
    return result;
  }



  // derived vector types -----------------------------------------------------
  template <class T>
  class numpy_vector
  : public boost::numeric::ublas::vector<T, numpy_array<T> >
  {
    private:
      typedef 
        boost::numeric::ublas::vector<T, numpy_array<T> >
        super;
    public:
      numpy_vector ()
      {}

      // observe that PyObject handles are implicitly convertible
      // to numpy_array
      numpy_vector(const numpy_array<T> &s)
        : super(s.size(), s)
      {
      }

      numpy_vector(int ndim_, const npy_intp *dims_)
        : super(size_from_dims(ndim_, dims_), 
            numpy_array<T>(ndim_, dims_))
      { }

      explicit 
        numpy_vector(typename super::size_type size)
        : super(size) 
        { }

      numpy_vector (
          typename super::size_type size, 
          const typename super::value_type &init)
        : super(size, init) 
      {}


      numpy_vector (const numpy_vector &v)
        : super(v)
      { }

      template<class AE>
      numpy_vector(const boost::numeric::ublas::vector_expression<AE> &ae)
      : super(ae)
      { }

      // numpy array metadata
      typename super::size_type ndim() const 
      { return this->data().ndim(); }
      const npy_intp *dims() const 
      { return this->data().dims(); }
      const npy_intp *strides() const 
      { return this->data().strides(); }
      npy_intp min_stride() const
      { return this->data().min_stride(); }
      npy_intp itemsize() const
      { return sizeof(T); }
      bool writable() const 
      { return this->data().writable(); }

      // several-d subscripts
      T &sub(npy_intp i) 
      { return *reinterpret_cast<T*>(PyArray_GETPTR1(this->data(), i)); }
      const T &sub(npy_intp i) const
      { return *reinterpret_cast<const T*>(PyArray_GETPTR1(this->data(), i)); }
      T &sub(npy_intp i, npy_intp j) 
      { return *reinterpret_cast<T*>(PyArray_GETPTR2(this->data(), i, j)); }
      const T &sub(npy_intp i, npy_intp j) const
      { return *reinterpret_cast<const T*>(PyArray_GETPTR2(this->data(), i, j)); }
      T &sub(npy_intp i, npy_intp j, npy_intp k) 
      { return *reinterpret_cast<T*>(PyArray_GETPTR3(this->data(), i, j, k)); }
      const T &sub(npy_intp i, npy_intp j, npy_intp k) const
      { return *reinterpret_cast<const T*>(PyArray_GETPTR3(this->data(), i, j, k)); }
      T &sub(npy_intp i, npy_intp j, npy_intp k, npy_intp l) 
      { return *reinterpret_cast<T*>(PyArray_GETPTR4(this->data(), i, j, k, l)); }
      const T &sub(npy_intp i, npy_intp j, npy_intp k, npy_intp l) const
      { return *reinterpret_cast<const T*>(PyArray_GETPTR4(this->data(), i, j, k, l)); }

      // shape manipulation 
      void reshape(int ndim_, const npy_intp *dims_, NPY_ORDER order=NPY_CORDER)
      {
        this->data().reshape(ndim_, dims_, order);
      }

      // as-strided accessor
      boost::numeric::ublas::vector_slice<numpy_vector>
        as_strided()
      {
        npy_intp ms = min_stride()/sizeof(T);
        return subslice(*this, 0, ms, this->size()/ms);
      }

      boost::numeric::ublas::vector_slice<const numpy_vector>
        as_strided() const
      {
        npy_intp ms = min_stride()/sizeof(T);
        return subslice(*this, 0, ms, this->size()/ms);
      }

      // as-ublas accessor
      super &as_ublas() 
      { return *this; }

      const super &as_ublas() const
      { return *this; }

      boost::python::handle<> to_python() const
      {
        return this->data().handle();
      }
  };




  // derived matrix types -----------------------------------------------------
  template<class T, class L/* = boost::numeric::ublas::row_major */> 
    /* default arg declared in forward decl */
  class numpy_matrix
  : public boost::numeric::ublas::matrix<T, L, numpy_array<T> > 
  {
    private:
      typedef 
        boost::numeric::ublas::matrix<T, L, numpy_array<T> >
        super;

    public:
      numpy_matrix ()
      { }

      numpy_matrix(
          typename super::size_type size1, 
          typename super::size_type size2)
      : super(size1, size2)
      { }

      numpy_matrix(
          typename super::size_type size1, 
          typename super::size_type size2, 
          const typename super::value_type &init)
      : super(size1, size2, init)
      { }

      numpy_matrix(
          typename super::size_type size1, 
          typename super::size_type size2, 
          const typename super::array_type &data)
      : super(size1, size2, data)
      { }

      // observe that PyObject handles are implicitly convertible
      // to numpy_array
      numpy_matrix(const typename super::array_type &data)
      : super(
          get_array_size1<typename super::orientation_category>(data),
          get_array_size2(data), 
          data)
      { }

      numpy_matrix(const numpy_matrix &m)
      : super(m)
      { }

      template<class AE>
      numpy_matrix (const boost::numeric::ublas::matrix_expression<AE> &ae)
      : super(ae)
      {
      }

      super &as_ublas() 
      { return *this; }

      const super &as_ublas() const
      { return *this; }

      boost::python::handle<> to_python() const
      { return matrix_to_python(*this); }
  };




  // data member treatment ----------------------------------------------------
  template <class T, class C>
  class by_value_rw_member_visitor 
  : public boost::python::def_visitor<by_value_rw_member_visitor<T, C> >
  {
    private:
      const char *m_name;
      T C::*m_member;
      const char *m_doc;

    public:
      by_value_rw_member_visitor(const char *name, T C::*member, const char *doc = 0)
        : m_name(name), m_member(member), m_doc(doc)
      { }

      template <class Class>
      void visit(Class& cl) const
      {
        cl.add_property(m_name, 
            boost::python::make_getter(m_member, 
              boost::python::return_value_policy<boost::python::return_by_value>()), 
            boost::python::make_setter(m_member), 
            m_doc);
      }
  };

  template <class T, class C>
  by_value_rw_member_visitor<T, C> by_value_rw_member(
      const char *name, T C::*member, const char *doc = 0)
  {
    return by_value_rw_member_visitor<T, C>(name, member, doc);
  }

  template <class T, class C>
  class by_value_ro_member_visitor 
  : public boost::python::def_visitor<by_value_ro_member_visitor<T, C> >
  {
    private:
      const char *m_name;
      T C::*m_member;
      const char *m_doc;

    public:
      by_value_ro_member_visitor(const char *name, T C::*member, const char *doc = 0)
        : m_name(name), m_member(member), m_doc(doc)
      { }

      template <class Class>
      void visit(Class& cl) const
      {
        cl.add_property(m_name, 
            make_getter(m_member, 
              boost::python::return_value_policy<boost::python::return_by_value>()), 
            m_doc);
      }
  };

  template <class T, class C>
  by_value_ro_member_visitor<T, C> by_value_ro_member(
      const char *name, T C::*member, const char *doc = 0)
  {
    return by_value_ro_member_visitor<T, C>(name, member, doc);
  }
}




// interaction with boost bindings --------------------------------------------
#ifdef PYUBLAS_HAVE_BOOST_BINDINGS

#include <boost/numeric/bindings/traits/ublas_vector.hpp>




namespace boost { namespace numeric { namespace bindings { namespace traits {
  template <typename T, typename V>
  struct vector_detail_traits< pyublas::numpy_array<T>, V > 
  : default_vector_traits< V, T > 
  {
#ifndef BOOST_NUMERIC_BINDINGS_NO_SANITY_CHECK
    BOOST_STATIC_ASSERT( 
        (boost::is_same< pyublas::numpy_array<T>, 
         typename boost::remove_const<V>::type >::value) );
#endif

    typedef pyublas::numpy_array<T>                      identifier_type; 
    typedef V                                            vector_type;
    typedef typename default_vector_traits<V,T>::pointer pointer;

    static pointer storage (vector_type& v) { return v.data(); }
  }; 
}}}}  

#endif




#endif
