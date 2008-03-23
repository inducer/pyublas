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

    if "PYLINEAR_CONF_TEMPLATE_VERSION" not in conf:
        old_config()

    if conf["PYLINEAR_CONF_TEMPLATE_VERSION"] < 2:
        old_config()

    # These are in Fortran. No headers available.
    conf["BLAS_INCLUDE_DIRS"] = []
    conf["LAPACK_INCLUDE_DIRS"] = []
    conf["ARPACK_INCLUDE_DIRS"] = []
    conf["DASKR_INCLUDE_DIRS"] = []
    conf["XERBLA_INCLUDE_DIRS"] = []

    INCLUDE_DIRS = ["src/cpp"] + \
                   conf["BOOST_INCLUDE_DIRS"]
    LIBRARY_DIRS = conf["BOOST_LIBRARY_DIRS"]
    LIBRARIES = conf["BPL_LIBRARIES"]

    OP_EXTRA_INCLUDE_DIRS = conf["BOOST_BINDINGS_INCLUDE_DIRS"]
    OP_EXTRA_LIBRARY_DIRS = []
    OP_EXTRA_LIBRARIES = []

    conf["USE_XERBLA"] = conf["SUPPLY_XERBLA"]
    conf["USE_BLAS"] = conf["HAVE_BLAS"]
    conf["USE_LAPACK"] = conf["HAVE_LAPACK"] and conf["HAVE_BLAS"]
    conf["USE_ARPACK"] = conf["HAVE_ARPACK"] and conf["USE_LAPACK"]
    conf["USE_UMFPACK"] = conf["USE_BLAS"] and conf["HAVE_UMFPACK"]
    conf["USE_DASKR"] = conf["USE_LAPACK"] and conf["HAVE_DASKR"]
    conf["USE_FFTW"] = conf["HAVE_FFTW"]

    if conf["HAVE_LAPACK"] and not conf["USE_LAPACK"]:
        print "*** LAPACK disabled because BLAS is missing"
    if conf["HAVE_ARPACK"] and not conf["USE_LAPACK"]:
        print "*** ARPACK disabled because LAPACK is not usable/missing"
    if conf["HAVE_UMFPACK"] and not conf["USE_UMFPACK"]:
        print "*** UMFPACK disabled because BLAS is missing"
    if conf["HAVE_DASKR"] and not conf["USE_DASKR"]:
        print "*** DASKR disabled because LAPACK is not usable/missing"

    OP_EXTRA_DEFINES = {}

    def handle_component(comp):
        if conf["USE_"+comp]:
            OP_EXTRA_DEFINES["USE_"+comp] = 1
            OP_EXTRA_INCLUDE_DIRS.extend(conf[comp+"_INCLUDE_DIRS"])
            OP_EXTRA_LIBRARY_DIRS.extend(conf[comp+"_LIBRARY_DIRS"])
            OP_EXTRA_LIBRARIES.extend(conf[comp+"_LIBRARIES"])

    handle_component("ARPACK")
    handle_component("UMFPACK")
    handle_component("FFTW")
    handle_component("DASKR")
    handle_component("LAPACK")
    handle_component("BLAS")
    handle_component("XERBLA")

    setup(name="PyUblas",
          version="0.90",
          description="Numpy-UBlas interoperability",
          author=u"Andreas Kloeckner",
          author_email="inform@tiker.net",
          license = "BSD",
          url="http://news.tiker.net/software/pyublas",
          packages=["pyublas"],
          package_dir={"pyublas": "src/python"},
          ext_package="pyublas",
          ext_modules=[ Extension("_internal", 
                                  [
                                      "src/wrapper/wrapper.cpp"
                                   ],
                                  include_dirs=INCLUDE_DIRS,
                                  library_dirs=LIBRARY_DIRS,
                                  libraries=LIBRARIES,
                                  extra_compile_args=conf["EXTRA_COMPILE_ARGS"],
                                  ),
                        ]
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
            cflags.append("-g")
            cvars['OPT'] = str.join(' ', cflags)
            cvars["CFLAGS"] = cvars["BASECFLAGS"] + " " + cvars["OPT"]
    # and now call main
    main()
