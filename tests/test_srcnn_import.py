"""Import and tiny CPU model checks; no checkpoint or image data required."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SRCNNImportTests(unittest.TestCase):
    def test_import_has_no_inference_side_effects_or_optional_dependencies(self):
        code = '''
import sys
from unittest.mock import patch
import torch
sys.modules['torchvision'] = None
sys.modules['pytorch_msssim'] = None
sys.modules['PIL'] = None
with patch('torch.load', side_effect=AssertionError('checkpoint read')), \\
     patch('os.makedirs', side_effect=AssertionError('directory creation')), \\
     patch('torch.cuda.is_available', side_effect=AssertionError('device probing')):
    from inference.srcnn_l1_inference import SRCNN
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_model_preserves_image_shape_on_cpu(self):
        sys.path.insert(0, str(ROOT))
        import torch
        from inference.srcnn_l1_inference import SRCNN
        torch.manual_seed(42)
        model = SRCNN().cpu().eval()
        with torch.no_grad():
            output = model(torch.randn(2, 1, 12, 16))
        self.assertEqual(output.shape, (2, 1, 12, 16))
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == '__main__':
    unittest.main()
