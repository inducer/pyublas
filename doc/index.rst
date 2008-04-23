Welcome to PyUblas's documentation!
===================================

PyUblas solves one main difficulty of developing hybrid numerical codes in
Python and C++: It integrates two major linear algebra libraries across the two
languages, namely `numpy <http://www.numpy.org>`_ and `Boost.Ublas
<http://www.boost.org/doc/libs/1_35_0/libs/numeric/ublas/doc/index.htm>`_. In Python, 
you are working with native numpy arrays, whereas in C++, PyUblas lets you work
with matrix and vector types immediately derived from and closely integrated with
Ublas.

PyUblas is built using and meant to be used with 
`Boost Python <http://www.boost.org/doc/libs/1_35_0/libs/numeric/ublas/doc/index.htm>`_.

PyUblas also has its own `web page
<http://mathema.tician.de/software/pyublas>`_.

Contents:

.. toctree::
    :maxdepth: 2

    installation  
    cpptypes
    converters
    wrapping
    pymodule
    faq

* :ref:`genindex`

.. TODO write about slice handling on conversion

