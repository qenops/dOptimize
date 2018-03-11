'''
    Test the functions to make sure they work
'''

import unittest
import dOptimize as do
import numpy as np

class TestPolynomialMethods(unittest.TestCase):

    def test_reflection(self):
        rd = np.array([0.,0.,1.])
        rp = np.array([0.,0.,0.])
        p = np.array([[0.,0.,-0.057],[0.,0.,0.],[-0.057,0.,5.]])
        fd, fp = do.reflectRayMVPolynomial(p, rd, rp)
        self.assertEqual(fp, np.array([0.,0.,5.]))

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()