"""File-pair discovery checks without decoding images or loading weights."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.srcnn_l1_inference import discover_image_names, main


class SRCNNPairTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lq, self.gt = self.root / 'lq', self.root / 'gt'
        self.lq.mkdir()
        self.gt.mkdir()

    def test_sorted_pairs_ignore_sidecars_and_subdirectories(self):
        for name in ['z.png', 'a.TIFF']:
            (self.lq / name).touch()
            (self.gt / name).touch()
        (self.lq / 'notes.txt').touch()
        (self.lq / 'nested.png').mkdir()
        (self.gt / 'unused.png').touch()
        self.assertEqual(discover_image_names(self.lq, self.gt), ['a.TIFF', 'z.png'])

    def test_missing_ground_truth_including_directory_is_reported(self):
        for name in ['a.png', 'b.png']:
            (self.lq / name).touch()
        (self.gt / 'b.png').mkdir()
        with self.assertRaisesRegex(FileNotFoundError, 'a.png, b.png'):
            discover_image_names(self.lq, self.gt)

    def test_absent_input_directory_is_reported(self):
        for lq, gt in [(self.root / 'missing', self.gt), (self.lq, self.root / 'missing')]:
            with self.subTest(lq=lq, gt=gt), self.assertRaises(NotADirectoryError):
                discover_image_names(lq, gt)

    def test_empty_input_fails_before_dependencies_weights_or_output(self):
        output = self.root / 'output'
        with patch.dict(sys.modules, {'torchvision': None, 'PIL': None}), \
             patch('inference.srcnn_l1_inference.load_model') as loader:
            with self.assertRaisesRegex(ValueError, 'No supported images'):
                main(['--lq-dir', str(self.lq), '--gt-dir', str(self.gt),
                      '--checkpoint', 'missing.pth', '--output-dir', str(output)])
        loader.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
