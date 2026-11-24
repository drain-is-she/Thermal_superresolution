"""CLI and synthetic checkpoint tests; no thermal dataset required."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.srcnn_l1_inference import SRCNN, load_model, main, parse_args


class SRCNNCLITests(unittest.TestCase):
    def test_explicit_paths_and_cpu_selection(self):
        args = parse_args(['--lq-dir', 'low', '--gt-dir', 'high', '--checkpoint', 'model.pth',
                           '--output-dir', 'results', '--device', 'cpu'])
        self.assertEqual((args.lq_dir, args.gt_dir, args.checkpoint, args.output_dir),
                         tuple(map(Path, ['low', 'high', 'model.pth', 'results'])))
        self.assertEqual(args.device, 'cpu')

    def test_missing_paths_fail_before_inference_dependencies(self):
        with patch.dict(sys.modules, {'torchvision': None, 'PIL': None}), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                main([])
        self.assertEqual(error.exception.code, 2)

    def test_synthetic_checkpoint_restores_cpu_weights_in_eval_mode(self):
        torch.manual_seed(42)
        original = SRCNN().cpu().eval()
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'model.pth'
            torch.save(original.state_dict(), checkpoint)
            with patch('torch.load', wraps=torch.load) as loader:
                restored = load_model(checkpoint, torch.device('cpu'))
            self.assertEqual(loader.call_args.kwargs['map_location'], torch.device('cpu'))
            self.assertTrue(loader.call_args.kwargs['weights_only'])
        self.assertFalse(restored.training)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
