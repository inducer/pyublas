#!/usr/bin/env python
# -*- coding: latin-1 -*-


def get_config_schema():
    from aksetup_helper import ConfigSchema, \
            BoostLibraries, Switch, StringListOption, make_boost_base_options

    return ConfigSchema(make_boost_base_options() + [
        BoostLibraries("python"),

        Switch("WITH_SPARSE_WRAPPERS", False, "Whether to build sparse wrappers"),
        Switch("USE_ITERATORS", False,
            "Whether to use iterators (faster, requires new Boost)"),

        StringListOption("CXXFLAGS", ["-Wno-sign-compare"],
            help="Any extra C++ compiler options to include"),
        StringListOption("LDFLAGS", [],
            help="Any extra linker options to include"),
        ])


def main():
    from aksetup_helper import hack_distutils, get_config, setup, \
            NumpyExtension

    hack_distutils()
    conf = get_config(get_config_schema())

    INCLUDE_DIRS = ["pyublas/include"] + conf["BOOST_INC_DIR"]
    LIBRARY_DIRS = conf["BOOST_LIB_DIR"]
    LIBRARIES = conf["BOOST_PYTHON_LIBNAME"]

    EXTRA_DEFINES = {}

    if conf["USE_ITERATORS"]:
        EXTRA_DEFINES["BOOST_UBLAS_USE_ITERATING"] = 1

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

    try:
        from distutils.command.build_py import build_py_2to3 as build_py
    except ImportError:
        # 2.x
        from distutils.command.build_py import build_py

    setup(
            name="PyUblas",
            version="2013.1",
            description="Seamless Numpy-UBlas interoperability",
            long_description=open("README.rst", "rt").read(),
            author="Andreas Kloeckner",
            author_email="inform@tiker.net",
            license="BSD",
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
                'Programming Language :: Python :: 2',
                'Programming Language :: Python :: 2.4',
                'Programming Language :: Python :: 2.5',
                'Programming Language :: Python :: 2.6',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.2',
                'Programming Language :: Python :: 3.3',
                'Programming Language :: C++',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Topic :: Office/Business',
                'Topic :: Utilities',
                ],

            # numpy is often under the setuptools radar.
            #setup_requires=[
                    #"numpy>=1.0.4",
                    #],
            install_requires=[
                    #"numpy>=1.0.4",
                    "pytest>=2",
                    ],

            packages=["pyublas"],
            ext_package="pyublas",
            ext_modules=[
                    NumpyExtension(
                        "_internal",
                        ext_src,
                        include_dirs=INCLUDE_DIRS,
                        library_dirs=LIBRARY_DIRS,
                        libraries=LIBRARIES,
                        define_macros=list(EXTRA_DEFINES.items()),
                        extra_compile_args=conf["CXXFLAGS"],
                        extra_link_args=conf["LDFLAGS"],
                        ),
                    NumpyExtension(
                        "testhelp_ext",
                        ["src/test/testhelp_ext.cpp"],
                        include_dirs=INCLUDE_DIRS,
                        library_dirs=LIBRARY_DIRS,
                        libraries=LIBRARIES,
                        define_macros=list(EXTRA_DEFINES.items()),
                        extra_compile_args=conf["CXXFLAGS"],
                        extra_link_args=conf["LDFLAGS"],
                        )
                    ],

            include_package_data=True,
            package_data={
                    "pyublas": [
                        "include/pyublas/*.hpp",
                        ]
                    },

            zip_safe=False,

            # 2to3 invocation
            cmdclass={'build_py': build_py})


if __name__ == '__main__':
    main()
