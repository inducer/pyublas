vars = [
    ("BOOST_INC_DIR", None,
        "Include directory for all of Boost C++"),
    ("BOOST_LIB_DIR", None,
        "Library directory for all of Boost C++"),
    ("BOOST_PYTHON_LIBNAME", "boost_python-gcc41-mt",
        "Name of the Boost Python library binary (without lib and .so)"),
    ("BOOST_BINDINGS_INC_DIR", None,
        "Include directory for the Boost bindings library"),
    # -------------------------------------------------------------------------
    ("HAVE_DASKR", False,
        "Whether to build with support for DASKR"),
    ("COMPILE_DASKR", False,
        "Whether to build DASKR. Implies HAVE_DASKR."),
    # -------------------------------------------------------------------------
    ("HAVE_BLAS", False,
        "Whether to build with support for BLAS"),
    ("BLAS_LIB_DIR", None,
        "Library directory for BLAS"),
    ("BLAS_LIB_NAMES", "blas",
        "Library names for BLAS, comma-separated"),
    # -------------------------------------------------------------------------
    ("HAVE_LAPACK", False,
        "Whether to build with support for LAPACK"),
    ("LAPACK_LIB_DIR", None,
        "Library directory for LAPACK"),
    ("LAPACK_LIB_NAMES", "lapack",
        "Library names for LAPACK, comma-separated"),
    # -------------------------------------------------------------------------
    ("HAVE_ARPACK", False,
        "Whether to build with support for ARPACK"),
    ("ARPACK_LIB_DIR", None,
        "Library directory for ARPACK"),
    ("ARPACK_LIB_NAMES", "arpack",
        "Library names for ARPACK, comma-separated"),
    # -------------------------------------------------------------------------
    ("HAVE_UMFPACK", False,
        "Whether to build with support for UMFPACK"),
    ("UMFPACK_INC_DIR", "/usr/include/suitesparse",
        "Include directory for UMFPACK"),
    ("UMFPACK_LIB_DIR", None,
        "Library directory for UMFPACK"),
    ("UMFPACK_LIB_NAMES", "umfpack,amd",
        "Library names for UMFPACK, comma-separated"),
    # -------------------------------------------------------------------------
    ("HAVE_FFTW", False,
        "Whether to build with support for FFTW3"),
    ("FFTW_INC_DIR", None,
        "Include directory for FFTW"),
    ("FFTW_LIB_DIR", None,
        "Library directory for FFTW"),
    ("FFTW_LIB_NAMES", "fftw3,m",
        "Library names for FFTW, comma-separated"),
    # -------------------------------------------------------------------------
    ("CXXFLAGS", "-Wno-sign-compare",
        "Any extra C++ compiler options to include"),
    ("SUPPLY_XERBLA", False,
        "Whether to compile and add our own XERBLA routine."
        "ATLAS LAPACK does not have one."),
    ]

subst_files = ["Makefile", "siteconf.py"]
