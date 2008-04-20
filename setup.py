#!/usr/bin/env python
# -*- coding: latin-1 -*-

import glob
import os
import os.path
import sys

def main():
    try:
        conf = {}
        execfile("siteconf.py", conf)
    except IOError:
        print "*** Please run configure first."
        sys.exit(1)

    from distutils.core import setup,Extension

    def old_config():
        print "*** You are using an old version of Pylinear's configuration."
        print "*** Please re-run configure."
        sys.exit(1)

    if "PYUBLAS_CONF_TEMPLATE_VERSION" not in conf:
        old_config()

    if conf["PYUBLAS_CONF_TEMPLATE_VERSION"] != 2:
        old_config()

    INCLUDE_DIRS = ["src/cpp"] + \
                   conf["BOOST_INCLUDE_DIRS"] + \
                   conf["NUMPY_INC_DIRS"]
    LIBRARY_DIRS = conf["BOOST_LIBRARY_DIRS"]
    LIBRARIES = conf["BPL_LIBRARIES"]

    EXTRA_INCLUDE_DIRS = conf["BOOST_BINDINGS_INCLUDE_DIRS"]
    EXTRA_LIBRARY_DIRS = []
    EXTRA_LIBRARIES = []

    EXTRA_DEFINES = {}

    ext_src = [
            "src/wrapper/main.cpp",
            "src/wrapper/converters.cpp",
            ] 

    if conf["WITH_SPARSE_WRAPPERS"]:
        ext_src += [
                "src/wrapper/sparse_build.cpp",
                "src/wrapper/sparse_execute.cpp",
                ]
        EXTRA_DEFINES["HAVE_SPARSE_WRAPPERS"] = 1

    setup(name="PyUblas",
          version="0.92",
          description="Seamless Numpy-UBlas interoperability",
          long_description="""
          PyUblas provides a seamless glue layer between
          `Numpy <http://www.numpy.org>`_ and
          `Boost.Ublas <http://www.boost.org/doc/libs/1_35_0/libs/numeric/ublas/doc/index.htm>`_
          for use with
          `Boost.Python <http://www.boost.org/doc/libs/1_35_0/libs/python/doc/index.html>`_.

          What does that mean? When writing 
          `hybrid scientific code <http://mathema.tician.de/node/455>`_, 
          one of the main problems is that abstractions that
          exist in the high-level language go away or become unwieldy in the
          low-level language. Sometimes libraries exist in both languages for
          these abstractions, but they refuse to talk to each other. PyUblas is
          a bridge between two such libraries, for some of the main
          abstractions used in scientific codes, namely vectors and matrices.

          Documentation
          =============

          See the 
          `PyUblas Documentation <http://mathema.tician.de/software/pyublas/doc>`_
          page.

          PyUblasExt
          ==========

          PyUblasExt is a companion to PyUblas and exposes a variety of useful
          additions to PyUblas, such as an "operator" class, matrix-free linear
          system solvers and eigensolvers. Interested? Head over to the
          `PyUblasExt <http://mathema.tician.de/software/pyublas/pyublasext>`_ 
          page.
          """,
          author=u"Andreas Kloeckner",
          author_email="inform@tiker.net",
          license = "BSD",
          url="http://mathema.tician.de/software/pyublas",
          classifiers=[
              'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Operating System :: MacOS :: MacOS X',
              'Operating System :: POSIX',
              'Programming Language :: Python',
              'Programming Language :: C++',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Office/Business',
              'Topic :: Utilities',
              'Topic :: Text Processing',
              ],

          packages=["pyublas"],
          package_dir={"pyublas": "src/python"},
          ext_package="pyublas",
          ext_modules=[ Extension("_internal", 
                                  ext_src,
                                  include_dirs=INCLUDE_DIRS+EXTRA_INCLUDE_DIRS,
                                  library_dirs=LIBRARY_DIRS+EXTRA_LIBRARY_DIRS,
                                  libraries=LIBRARIES+EXTRA_LIBRARIES,
                                  define_macros=EXTRA_DEFINES.items(),
                                  extra_compile_args=conf["EXTRA_COMPILE_ARGS"],
                                  ),
                        ],
          data_files=[("include/pyublas", glob.glob("src/cpp/pyublas/*.hpp"))],
         )




if __name__ == '__main__':
    # hack distutils.sysconfig to eliminate debug flags
    # stolen from mpi4py
    import sys
    if not sys.platform.lower().startswith("win"):
        from distutils import sysconfig

        cvars = sysconfig.get_config_vars()
        cflags = cvars.get('OPT')
        if cflags:
            cflags = cflags.split()
            for bad_prefix in ('-g', '-O', '-Wstrict-prototypes'):
                for i, flag in enumerate(cflags):
                    if flag.startswith(bad_prefix):
                        cflags.pop(i)
                        break
                if flag in cflags:
                    cflags.remove(flag)
            cflags.append("-O3")
            cvars['OPT'] = str.join(' ', cflags)
            cvars["CFLAGS"] = cvars["BASECFLAGS"] + " " + cvars["OPT"]
    # and now call main
    main()
