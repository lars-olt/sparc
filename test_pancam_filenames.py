import asdf
import unittest

from sparc.utils.pancam_helpers import parse_pcam_fn


class PancamFilenameTests(unittest.TestCase):
    def test_parse_pcam_fn_accepts_archive_location_placeholders(self):
        for location in ('as__', 'as##'):
            with self.subTest(location=location):
                parsed = parse_pcam_fn(
                    f'2p228988949iof{location}p2580l2a1.img'
                )

                self.assertIsNotNone(parsed)
                self.assertEqual(parsed['SEQ_ID'], 'p2580')
                self.assertEqual(parsed['FILTER'], 'l2')


if __name__ == '__main__':
    unittest.main()
