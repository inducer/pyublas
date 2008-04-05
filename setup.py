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

    if conf["PYUBLAS_CONF_TEMPLATE_VERSION"] != 1:
        old_config()

    INCLUDE_DIRS = ["src/cpp"] + \
                   conf["BOOST_INCLUDE_DIRS"]
    LIBRARY_DIRS = conf["BOOST_LIBRARY_DIRS"]
    LIBRARIES = conf["BPL_LIBRARIES"]

    EXTRA_INCLUDE_DIRS = conf["BOOST_BINDINGS_INCLUDE_DIRS"]
    EXTRA_LIBRARY_DIRS = []
    EXTRA_LIBRARIES = []

    OP_EXTRA_DEFINES = {}

    setup(name="PyUblas",
          version="0.90",
          description="Seamless Numpy-UBlas interoperability",
          author=u"Andreas Kloeckner",
          author_email="inform@tiker.net",
          license = "BSD",
          url="http://news.tiker.net/software/pyublas",
          packages=["pyublas"],
          package_dir={"pyublas": "src/python"},
          ext_package="pyublas",
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
          ext_modules=[ Extension("_internal", 
                                  [
                                      "src/wrapper/main.cpp",
                                      "src/wrapper/converters.cpp",
                                      "src/wrapper/sparse_build.cpp",
                                      "src/wrapper/sparse_execute.cpp"
                                   ],
                                  include_dirs=INCLUDE_DIRS+EXTRA_INCLUDE_DIRS,
                                  library_dirs=LIBRARY_DIRS+EXTRA_LIBRARY_DIRS,
                                  libraries=LIBRARIES+EXTRA_LIBRARIES,
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
