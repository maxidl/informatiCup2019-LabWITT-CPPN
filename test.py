from pathlib import Path
import numpy as np
import util
import cppn_init
import warnings
import unittest

cwd = Path('.')
test_img_dir = cwd / 'test_images'


class TestUtilFunctions(unittest.TestCase):

    def check_test_img_exists(self):
        self.assertTrue((test_img_dir / '0.png').exists())
        self.assertTrue((test_img_dir / '0.png').is_file())

    def test_img_save_and_load(self):
        test_img = util.load_image(test_img_dir / '2.png', size=64)
        self.assertEqual(type(test_img), np.ndarray)
        self.assertEqual(test_img.shape, (64, 64, 3))
        self.assertEqual(test_img.dtype, np.float64)

        util.save_image(test_img_dir / 'test.png', test_img)
        file = (test_img_dir / 'test.png')
        self.assertTrue(file.exists())
        self.assertTrue(file.is_file())
        self.assertTrue(np.allclose(test_img, util.load_image(file)))
        file.unlink()

    # TODO test gif saving

    def test_api_querying(self):
        self.assertEqual(util._URL, 'https://phinau.de/trasi')
        test_img = util.load_image(test_img_dir / '2.png', size=64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            confidence = util.get_confidence(test_img)
        self.assertEqual(type(confidence), np.float64)
        self.assertEqual(confidence, 0.99997401)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            confidence = util.get_confidence(test_img, target_class='Zulässige Höchstgeschwindigkeit (30)')
        self.assertEqual(confidence, 0.00002541)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            confidence = util.get_confidence(test_img,
                                             'Schnee- oder Eisglätte')
        self.assertEqual(confidence, 1e-10)


class TestCPPN(unittest.TestCase):

    def test_similarity(self):
        test_img = util.load_image(test_img_dir / '2.png')
        test_img2 = np.clip(test_img + 0.1, 0.0, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sim1 = cppn_init.similarity(test_img, test_img, True)
            sim2 = cppn_init.similarity(test_img, test_img2, True)
            self.assertEqual(sim1, 1.0)
            self.assertLess(sim2, 1.0)

    def test_mutation(self):
        weights = np.random.rand(100)
        self.assertEqual(weights.shape, cppn_init.mutate(weights, 0.5, 0.05).shape)

    def test_cppn(self):
        cppn = cppn_init.init_cppn_from_img(util.load_image(test_img_dir / '2.png'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertTrue(cppn)
        self.assertEqual(cppn.render_image().shape, (64, 64, 3))
        weights = cppn.get_weights()
        cppn.set_weights(weights)
        cppn.reset()
        cppn.render_image()


if __name__ == '__main__':
    unittest.main()
