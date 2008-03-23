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




#ifndef _AFAYFYDASDFAH_PYUBLAS_HEADER_SEEN_PYTHON_HELPERS_HPP
#define _AFAYFYDASDFAH_PYUBLAS_HEADER_SEEN_PYTHON_HELPERS_HPP




#define PYUBLAS_PYERROR(TYPE, REASON) \
{ \
  PyErr_SetString(PyExc_##TYPE, REASON); \
  throw boost::python::error_already_set(); \
}




namespace pyublas
{

}




#endif

