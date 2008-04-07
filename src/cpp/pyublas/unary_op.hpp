//
// Copyright (c) 2004-2006
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




#ifndef HEADER_SEEN_PYUBLAS_UNARY_OP_HPP
#define HEADER_SEEN_PYUBLAS_UNARY_OP_HPP




#include <cmath>
#include <boost/numeric/ublas/functional.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>




namespace pyublas {
  template <class Functor>
  struct unary_op
  {
    template<class E> 
    static
    typename boost::numeric::ublas::vector_unary_traits<E, Functor>::result_type
    apply(const boost::numeric::ublas::vector_expression<E> &e) 
    {
      typedef typename boost::numeric::ublas::vector_unary_traits<E, Functor>::expression_type expression_type;
      return expression_type (e ());
    }

    template<class E>
    static
    typename boost::numeric::ublas::matrix_unary1_traits<E, Functor>::result_type
    apply(const boost::numeric::ublas::matrix_expression<E> &e) 
    {
      typedef typename boost::numeric::ublas::matrix_unary1_traits<E, Functor>::expression_type expression_type;
      return expression_type (e ());
    }
  };

  namespace unary_ops
  {
    struct fabs
    {
      typedef double value_type;
      typedef const double &argument_type;
      typedef double result_type;

      static result_type apply(argument_type x)
      {
        return ::std::fabs(x);
      }
    };
  }
}




#endif
