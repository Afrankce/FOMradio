import unittest
from manifest import config_for, beam_index


class ManifestTests(unittest.TestCase):
    def test_same_angle_uses_different_beam_indices(self):
        self.assertEqual([beam_index(8, a) for a in (-14, 0, 14)], [2, 4, 6])
        self.assertEqual([beam_index(32, a) for a in (-14, 0, 14)], [18, 32, 46])

    def test_reject_unpublished_query(self):
        with self.assertRaises(ValueError): beam_index(8, 1)
        with self.assertRaises(ValueError): beam_index(16, 0)

    def test_config(self):
        self.assertEqual(config_for(8), 'freq_6.7GHz_64TR_8beams_pattern_tr38901')


if __name__ == '__main__': unittest.main()
