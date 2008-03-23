//
// Copyright (c) 2008
// Andreas Kloeckner
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




#include <complex>
#include <pyublas/python_helpers.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/python.hpp>
#include <boost/foreach.hpp>
#include <numpy/arrayobject.h>





namespace pyublas
{
  NPY_TYPES get_typenum(bool) { return NPY_BOOL; }
  NPY_TYPES get_typenum(npy_bool) { return NPY_BOOL; }
  NPY_TYPES get_typenum(npy_byte) { return NPY_BYTE; }
  NPY_TYPES get_typenum(npy_ubyte) { return NPY_UBYTE; }
  NPY_TYPES get_typenum(npy_short) { return NPY_SHORT; }
  NPY_TYPES get_typenum(npy_ushort) { return NPY_USHORT; }
  NPY_TYPES get_typenum(npy_int) { return NPY_INT; }
  NPY_TYPES get_typenum(npy_uint) { return NPY_UINT; }
  NPY_TYPES get_typenum(npy_long) { return NPY_LONG; }
  NPY_TYPES get_typenum(npy_ulong) { return NPY_ULONG; }
  NPY_TYPES get_typenum(npy_longlong) { return NPY_LONGLONG; }
  NPY_TYPES get_typenum(npy_ulonglong) { return NPY_ULONGLONG; }
  NPY_TYPES get_typenum(npy_float) { return NPY_FLOAT; }
  NPY_TYPES get_typenum(npy_double) { return NPY_DOUBLE; }
  NPY_TYPES get_typenum(npy_longdouble) { return NPY_LONGDOUBLE; }
  NPY_TYPES get_typenum(npy_cfloat) { return NPY_CFLOAT; }
  NPY_TYPES get_typenum(npy_cdouble) { return NPY_CDOUBLE; }
  NPY_TYPES get_typenum(npy_clongdouble) { return NPY_CLONGDOUBLE; }
  NPY_TYPES get_typenum(std::complex<float>) { return NPY_CFLOAT; }
  NPY_TYPES get_typenum(std::complex<double>) { return NPY_CDOUBLE; }
  NPY_TYPES get_typenum(std::complex<long double>) { return NPY_CLONGDOUBLE; }
  NPY_TYPES get_typenum(object) { return NPY_OBJECT; }
  NPY_TYPES get_typenum(handle<>) { return NPY_OBJECT; }
  /* NPY_STRING, NPY_UNICODE unsupported for now */

  template <class T>
  class numpy_storage<T> 
  : public boost::numeric::ublas::carray_adaptor<T>
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
      numpy_storage()
      {
      }

      numpy_storage(size_type n)
      {
        npy_intp dims[] = { n };
        m_numpy_array = handle<>(
            PyArray_SimpleNew(1, dims, get_typenum(T())));
      }

      numpy_storage(size_type n, const value_type &v)
      {
        if (size)
        {
          npy_intp dims[] = { n };
          m_numpy_array = handle<>(
              PyArray_SimpleNew(1, dims, get_typenum(T())));
          fill(begin(), end(), v);
        }
      }

      numpy_storage(size_type n, const value_type &v)
      {
        if (size)
        {
          npy_intp dims[] = { n };
          m_numpy_array = handle<>(
              PyArray_SimpleNew(1, dims, get_typenum(T())));
          fill(begin(), end(), v);
        }
      }

      /* MISSING
      Range constructor 	X(i, j) 	i and j are Input Iterators whose value type is convertible to T 	X
      */

      numpy_storage(const boost::python::handle<> &obj)
        : m_numpy_array(obj)
      {
        if (!PyArray_Check(obj.get()))
          PYUBLAS_PYERROR(TypeError, 
              "argument is not a numpy array");
        if (!PyArray_CHECKFLAGS(obj, NPY_ALIGNED | NPY_CONTIGUOUS)
            "argument array is not aligned and contiguous");
      }

      // Resizing
      void resize(size_type size) 
      {
        throw std::runtime_error("numpy storage is not (yet) resizable");
      }

      void resize(size_type size, value_type init) 
      {
          throw std::runtime_error("numpy storage is not (yet) resizable");
      }

      size_type size() const 
      {
        return PyArray_SIZE(m_numpy_array.get());
      }

      bool writable() const
      {
        return PyArray_CHECKFLAGS(obj, NPY_WRITABLE);
      }

    private:
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

    public:
      // Element access
      const_reference operator [] (size_type i) const 
      {
        BOOST_UBLAS_CHECK (i < size_, bad_index ());
        return data()[i];
      }

      reference operator [] (size_type i) 
      {
        BOOST_UBLAS_CHECK (i < size_, bad_index ());
        return data()[i];
      }

      // Assignment
      numpy_borrowed_storage &operator=(
          const numpy_borrowed_storage &a) 
      {
        m_numpy_array = a.m_numpy_array;
      }

      numpy_borrowed_storage &assign_temporary(
          numpy_borrowed_storage &a) 
      {
        m_numpy_array = a.m_numpy_array;
      }

        // Swapping
      void swap (numpy_borrowed_storage &a) 
      {
        if (this != &a)
          std::swap(m_numpy_array, a.m_numpy_array);
      }

      friend void swap(
          numpy_borrowed_storage &a1, 
          shallow_array_adaptor &a2) 
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

      const_reverse_iterator rbegin () const 
      {
        return std::const_reverse_iterator (end ());
      }

      const_reverse_iterator rend () const 
      {
        return std::const_reverse_iterator (begin ());
      }

      reverse_iterator rbegin () 
      {
        return std::reverse_iterator (end ());
      }

      reverse_iterator rend () 
      {
        return std::reverse_iterator (begin ());
      }
  };




  template <class T>
  class numpy_vector<T> 
  : public boost::numeric::ublas::vector<T, numpy_storage<T> >
    {
      private:
        typedef 
          boost::numeric::ublas::vector<T, numpy_storage<T> >
          super;
      public:
        numpy_vector ()
        {}

        // observe that PyObject handles are implicitly convertible
        // to numpy_storage
        numpy_vector(numpy_storage &s)
          : super(s.size(), s)
        {
        }

        explicit 
        numpy_vector (size_type size)
        : super(size) 
        { }

        numpy_vector (size_type size, const value_type &init)
          : data_ (size, init) 
        {}

        numpy_vector (const vector &v):
            vector_container<self_type> (),
            data_ (v.data_) {}

        template<class AE>
        BOOST_UBLAS_INLINE
        vector (const vector_expression<AE> &ae)
        : super(numpy_storage(ae.size()))
        {
          vector_assign<scalar_assign> (*this, ae);
        }
    };
}




#endif
