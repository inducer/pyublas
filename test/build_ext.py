#! /usr/bin/env python

def build_ext():
    import sys
    sys.path.append("..")
    from setup import get_config_schema
    schema = get_config_schema()
    schema.set_conf_dir("..")

    from aksetup_helper import get_config

    conf = get_config(schema)

    import distutils.sysconfig
    import numpy
    inc_paths = [
        "../src/cpp", 
        distutils.sysconfig.get_python_inc(),
        numpy.get_include(),
        ] + conf["BOOST_INC_DIR"]
    lib_paths = conf["BOOST_LIB_DIR"]
    libs = conf["BOOST_PYTHON_LIBNAME"]

    import os
    cmd = ("c++ -g -fpic -shared %s %s %s test_ext.cpp -o test_ext.so"
            % (
                " ".join("-I"+ip for ip in inc_paths),
                " ".join("-L"+ip for ip in lib_paths),
                " ".join("-l"+ip for ip in libs),
                ))
    print cmd
    os.system(cmd)



if __name__ == "__main__":
    build_ext()
