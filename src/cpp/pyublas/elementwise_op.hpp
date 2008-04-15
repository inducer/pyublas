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




#ifndef HEADER_SEEN_PYUBLAS_ELEMENTWISE_OP_HPP
#define HEADER_SEEN_PYUBLAS_ELEMENTWISE_OP_HPP




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
      typedef typename boost::numeric::ublas::vector_unary_traits<E, Functor>
        ::expression_type expression_type;
      return expression_type(e());
    }

    template<class E>
    static
    typename boost::numeric::ublas::matrix_unary1_traits<E, Functor>::result_type
    apply(const boost::numeric::ublas::matrix_expression<E> &e) 
    {
      typedef typename boost::numeric::ublas::matrix_unary1_traits<E, Functor>
        ::expression_type expression_type;
      return expression_type(e());
    }
  };




  template <class Functor>
  struct binary_op
  {
    template<class E1, class E2> 
    static
    typename boost::numeric::ublas::vector_binary_traits<E1, E2, Functor>::result_type
    apply(
        const boost::numeric::ublas::vector_expression<E1> &e1,
        const boost::numeric::ublas::vector_expression<E2> &e2
        ) 
    {
      typedef typename boost::numeric::ublas::vector_binary_traits<E1, E2, Functor> 
        ::expression_type expression_type;
      return expression_type(e1(), e2());
    }

    template<class E1, class E2> 
    static
    typename boost::numeric::ublas::matrix_binary_traits<E1, E2, Functor>::result_type
    apply(
        const boost::numeric::ublas::matrix_expression<E1> &e1,
        const boost::numeric::ublas::matrix_expression<E2> &e2
        ) 
    {
      typedef typename boost::numeric::ublas::matrix_binary_traits<E1, E2, Functor> 
        ::expression_type expression_type;
      return expression_type(e1(), e2());
    }
  };




  namespace unary_ops
  {
    class fabs : public boost::numeric::ublas::scalar_real_unary_functor<double>
    {
      private:
        typedef boost::numeric::ublas::scalar_unary_functor<double> super;

      public:
        static super::result_type apply(super::argument_type x)
        {
          return ::std::fabs(x);
        }
    };
  }
  


  
  namespace binary_ops
  {
    template<class T1, class T2=T1>
    class max : public boost::numeric::ublas::scalar_binary_functor<T1, T2> 
    {
      private:
        typedef boost::numeric::ublas::scalar_binary_functor<T1, T2> super;

      public:
        static typename super::result_type apply(
            typename super::argument1_type t1, 
            typename super::argument1_type t2) 
        {
          if (t1 >= t2)
            return t1;
          else
            return t2;
        }
    };




    template<class T1, class T2=T1>
    class min : public boost::numeric::ublas::scalar_binary_functor<T1, T2> 
    {
      private:
        typedef boost::numeric::ublas::scalar_binary_functor<T1, T2> super;

      public:
        static typename super::result_type apply(
            typename super::argument1_type t1, 
            typename super::argument1_type t2) 
        {
          if (t1 <= t2)
            return t1;
          else
            return t2;
        }
    };
  }
}




#endif
