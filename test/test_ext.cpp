#include <algorithm>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
//#include <pyublas/numpy.hpp>
#include <boost/foreach.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>




using namespace boost::python;
// using namespace pyublas;
namespace ublas = boost::numeric::ublas;




template <class T>
T doublify(T x)
{
  return 2*x;
}




template <class T>
object doublify_list_of(object o)
{
  std::vector<T> lst;
  std::copy(
      stl_input_iterator<T>(o),
      stl_input_iterator<T>(),
      std::back_inserter(lst));

  list result;
  BOOST_FOREACH(T x, lst)
    result.append(2*x);
  return result;
}




BOOST_PYTHON_MODULE(test_ext)
{
  def("dbl_int", doublify<unsigned int>);
  def("dbl_float", doublify<double>);
  def("dbl_list_int", doublify_list_of<unsigned int>);
  def("dble_list_float", doublify_list_of<double>);

  /*
  def("dbl_numpy_mat", doublify<numpy_matrix<double> >);
  def("dbl_numpy_mat_cm", 
      doublify<numpy_matrix<double, ublas::column_major> >);
  def("dbl_numpy_vec", 
      doublify<numpy_vector<double> >);
      */
  def("dbl_ublas_vec", 
      doublify<ublas::vector<double> >);
  def("dbl_ublas_mat", 
      doublify<ublas::matrix<double> >);
}
