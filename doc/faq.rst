Frequently Asked Questions
==========================

Where do the headers get installed?
-----------------------------------

Its easiest to find this location using this Python code snippet::

    from imp import find_module
    file, pathname, descr = find_module("pyublas")
    from os.path import join
    return join(pathname, "..", "include")

Since PyUblas requires numpy, you'll need a similar snippet for that, too::

    from imp import find_module
    file, pathname, descr = find_module("numpy")
    from os.path import join
    return join(pathname, "core", "include")

What about 0-dimensional arrays?
--------------------------------

0-dimensional arrays are supported by PyUblas, they get converted to
:ctype:`numpy_vector` instances of length 1.
