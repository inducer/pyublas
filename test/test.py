def main():
    try:
        import test_ext as te
    except ImportError:
        print "-----------------------------------------------------"
        print "You need to compile the test extension."
        print "Adapt and run ./build_ext.sh before running this."
        print "-----------------------------------------------------"
        raise

    import numpy
    #import pyublas
    u = numpy.int32(5)
    print te.dbl_list_int([u])
    #zero_d = numpy.zeros((), dtype=numpy.int32)
    #zero_d[()] = 17
    #print te.dbl_numpy_vec(zero_d)
    a = numpy.ones((5,5))
    print te.dbl_numpy_mat(a)

if __name__ == "__main__":
    main()
