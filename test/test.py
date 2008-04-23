import pyublas
import numpy
import unittest

try:
    import test_ext as te
except ImportError:
    print "------------------------------------------------"
    print "You need to compile the test extension."
    print "Run ./build_ext.py to do so."
    print "I'll try doing this for you, but no guarantees:"
    print "------------------------------------------------"
    from build_ext import build_ext
    build_ext()
    import test_ext as te





class TestMatrices(unittest.TestCase):
    def etest_array_scalars(self):
        u = numpy.int32(5)
        assert te.dbl_int(u) == 10

    def test_regular_vector(self):
        a = numpy.ones((5,), dtype=float)
        assert (te.dbl_numpy_vec(a) == 2*a).all()

    def test_0d_array(self):
        a = numpy.ones((), dtype=float)

        # 0d converted to 1d, single element
        dbl_1 = te.dbl_numpy_vec(a)
        assert dbl_1.shape == (1,)
        assert dbl_1[0] == 2

        # everything preserved
        dbl_2 = te.dbl_numpy_vec_2(a)
        assert dbl_2.shape == ()
        assert dbl_2[()] == 2

        # everything preserved
        dbl_3 = te.dbl_numpy_vec_3(a)
        assert dbl_3.shape == ()
        assert dbl_3[()] == 2

    def test_vec_slice_noncontig(self):
        a_orig = numpy.ones((10,), dtype=float)
        a = a_orig[::2]

        # whole vector multiplied (!) -> noncontiguous slice
        dbl_1 = te.dbl_numpy_vec(a)
        assert dbl_1.shape == (10,)
        assert (dbl_1 == 2*a_orig).all()

        # only slice multiplied
        dbl_2 = te.dbl_numpy_vec_2(a)
        assert dbl_2.shape == (5,)
        assert (dbl_2 == 2*a).all()

        # same here
        dbl_3 = te.dbl_numpy_vec_3(a)
        assert dbl_3.shape == (5,)
        assert (dbl_3 == 2*a).all()

        # when copying, take strides into account
        dbl_4 = te.dbl_ublas_vec(a)
        assert dbl_4.shape == (5,)
        assert (dbl_4 == 2*a).all()

    def test_vec_slice_noncontig_inplace(self):
        a_orig = numpy.ones((10,), dtype=float)
        a = a_orig[::2]

        te.dbl_numpy_vec_inplace(a)
        assert (a_orig[::2] == 2).all()
        assert (a_orig[1::2] == 1).all()

    def test_vec_slice_contig_inplace(self):
        a_orig = numpy.ones((10,), dtype=float)
        a = a_orig[4:7]

        te.dbl_numpy_vec_inplace(a)
        assert (a_orig[4:7] == 2).all()
        assert (a_orig[0:4] == 1).all()
        assert (a_orig[7:10] == 1).all()

    def test_matrix(self):
        a = numpy.ones((10,10), dtype=float)

        dbl_0 = te.dbl_numpy_mat(a)
        assert dbl_0.shape == (10,10)
        assert (dbl_0 == 2).all()

        dbl_1 = te.dbl_ublas_mat(a)
        assert dbl_1.shape == (10,10)
        assert (dbl_1 == 2).all()

    def test_matrix_slice(self):
        a = numpy.ones((10,10), dtype=float)

        dbl_0 = te.dbl_numpy_mat(a[3:7])
        assert dbl_0.shape == (4,10)
        assert (dbl_0 == 2).all()

        try:
            dbl_0 = te.dbl_numpy_mat(a[:, 3:7])
            assert False
        except TypeError:
            pass

    def test_matrix_slice_inplace(self):
        a = numpy.ones((10,10), dtype=float)

        te.dbl_numpy_mat_inplace(a[3:7])
        assert a.sum() == 140

        try:
            te.dbl_numpy_mat_inplace(a[:, 3:7])
            assert False
        except TypeError:
            pass




if __name__ == "__main__":
    unittest.main()
