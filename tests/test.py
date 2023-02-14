import unittest
import numpy as np
from TNAgen import Generator

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.gen = Generator()

    def test_gen_returns_correct_num(self):
        n_images = 10
        images, labels = self.gen.generate('Koi_Fish', n_images)
        self.assertEqual(len(images), n_images)
        self.assertEqual(len(labels), n_images)

    def test_gen_output_shape(self):
        n_images = 10
        images, labels = self.gen.generate('Koi_Fish', n_images)
        self.assertEqual(images.shape, (n_images, 140, 170))

    def test_gen_all_returns_correct_num(self):
        n_images = 2
        images, labels = self.gen.generate_all(n_images)
        self.assertEqual(len(images), len(self.gen.glitches)*n_images)
        self.assertEqual(len(labels), len(self.gen.glitches)*n_images)

    def test_gen_all_output_shape(self):
        n_images = 2
        images, labels = self.gen.generate_all(n_images)
        self.assertEqual(images.shape, (len(self.gen.glitches)*n_images, 140, 170))
