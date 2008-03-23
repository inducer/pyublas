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
    ("CXXFLAGS", "-Wno-sign-compare",
        "Any extra C++ compiler options to include"),
    ]

subst_files = ["Makefile", "siteconf.py"]
