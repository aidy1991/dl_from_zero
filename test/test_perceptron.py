import unittest
import numpy as np
from src.perceptron import AND, OR, NAND, XOR


class TestPerceptron(unittest.TestCase):
    def test_and(self):
        self.assertEqual(AND(np.array([0, 0])), 0)
        self.assertEqual(AND(np.array([0, 1])), 0)
        self.assertEqual(AND(np.array([1, 0])), 0)
        self.assertEqual(AND(np.array([1, 1])), 1)

    def test_or(self):
        self.assertEqual(OR(np.array([0, 0])), 0)
        self.assertEqual(OR(np.array([0, 1])), 1)
        self.assertEqual(OR(np.array([1, 0])), 1)
        self.assertEqual(OR(np.array([1, 1])), 1)

    def test_nand(self):
        self.assertEqual(NAND(np.array([0, 0])), 1)
        self.assertEqual(NAND(np.array([0, 1])), 1)
        self.assertEqual(NAND(np.array([1, 0])), 1)
        self.assertEqual(NAND(np.array([1, 1])), 0)

    def test_xor(self):
        self.assertEqual(XOR(np.array([0, 0])), 0)
        self.assertEqual(XOR(np.array([0, 1])), 1)
        self.assertEqual(XOR(np.array([1, 0])), 1)
        self.assertEqual(XOR(np.array([1, 1])), 0)


if __name__ == '__main__':
    unittest.main()
