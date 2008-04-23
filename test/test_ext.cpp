#include <iostream>
#include <algorithm>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <pyublas/numpy.hpp>
#include <boost/foreach.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>




using namespace boost::python;
using namespace pyublas;
namespace ublas = boost::numeric::ublas;




template <class T>
T doublify(T x)
{
  return 2*x;
}




template <class T>
void double_inplace(T x)
{
  x *= 2;
}




/* The following two versions prerserve shape on output: */
template <class T>
numpy_vector<T> doublify_numpy_vector_2(numpy_vector<T> x)
{
  numpy_vector<T> result(x.ndim(), x.dims());
  result.assign(2*x.as_strided());
  return result;
}




template <class T>
numpy_vector<T> doublify_numpy_vector_3(numpy_vector<T> x)
{
  numpy_vector<T> result(2*x.as_strided());
  result.reshape(x.ndim(), x.dims());
  return result;
}




template <class T>
void double_numpy_vector_inplace(numpy_vector<T> x)
{
  x.as_strided() *= 2;
}




BOOST_PYTHON_MODULE(test_ext)
{
  def("dbl_int", doublify<int>);
  def("dbl_float", doublify<double>);

  def("dbl_numpy_mat", doublify<numpy_matrix<double> >);
  def("dbl_numpy_mat_cm", 
      doublify<numpy_matrix<double, ublas::column_major> >);

  def("dbl_numpy_mat_inplace", double_inplace<numpy_matrix<double> >);
  def("dbl_numpy_mat_cm_inplace", 
      double_inplace<numpy_matrix<double, ublas::column_major> >);

  def("dbl_numpy_vec", 
      doublify<numpy_vector<double> >);
  def("dbl_numpy_vec_2", 
      doublify_numpy_vector_2<double>);
  def("dbl_numpy_vec_3", 
      doublify_numpy_vector_3<double>);
  def("dbl_numpy_vec_inplace", 
      double_numpy_vector_inplace<double>);
  def("dbl_ublas_vec", 
      doublify<ublas::vector<double> >);
  def("dbl_ublas_mat", 
      doublify<ublas::matrix<double> >);
}
