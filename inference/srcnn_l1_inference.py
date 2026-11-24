import torch
import torch.nn as nn
import os
import argparse
from pathlib import Path

# ---------------- MODEL ----------------
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):  
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# ---------------- METRICS ----------------
def denormalize(x):
    return (x * 0.5) + 0.5

def calculate_psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100
    return 10 * torch.log10(1.0 / mse)

def calculate_ssim(sr, hr):
    from pytorch_msssim import ssim

    return ssim(sr, hr, data_range=1.0, size_average=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate an SRCNN L1 checkpoint on paired thermal images.")
    parser.add_argument("--lq-dir", type=Path, required=True, help="Low-resolution image directory")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Ground-truth directory with matching filenames (448 x 640)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SRCNN state dictionary")
    parser.add_argument("--output-dir", type=Path, default=Path("srcnn_l1_output"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def load_model(checkpoint, device):
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.eval()


def main(argv=None):
    args = parse_args(argv)
    from torchvision import transforms
    from PIL import Image

    # ---------------- PATHS ----------------
    lq_dir = args.lq_dir
    gt_dir = args.gt_dir

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)


    # ---------------- TRANSFORMS ----------------
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    upsample = transforms.Resize(
        (448, 640),
        interpolation=transforms.InterpolationMode.BICUBIC
    )

    to_pil = transforms.ToPILImage()


    # ---------------- LOAD MODEL ----------------
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = load_model(args.checkpoint, device)


    # ---------------- INFERENCE ----------------
    image_names = sorted(os.listdir(lq_dir))

    total_psnr = 0
    total_ssim = 0

    with torch.no_grad():
        for name in image_names:

            lr_path = os.path.join(lq_dir, name)
            gt_path = os.path.join(gt_dir, name)

            # load images
            lr = Image.open(lr_path).convert("L")
            gt = Image.open(gt_path).convert("L")

            # preprocess
            lr = upsample(lr)
            lr = normalize(to_tensor(lr)).unsqueeze(0).to(device)

            gt = to_tensor(gt).unsqueeze(0).to(device)

            # inference
            sr = model(lr)

            # denormalize
            sr = denormalize(sr)
            sr = torch.clamp(sr, 0, 1)

            # ---- METRICS ----
            total_psnr += calculate_psnr(sr, gt)
            total_ssim += calculate_ssim(sr, gt)

            # ---- SAVE IMAGE ----
            sr_img = sr.squeeze(0).cpu()
            sr_img = to_pil(sr_img)

            save_path = os.path.join(output_dir, name)
            sr_img.save(save_path)


    # ---------------- FINAL RESULTS ----------------
    avg_psnr = total_psnr / len(image_names)
    avg_ssim = total_ssim / len(image_names)

    print("\n📊 FINAL RESULTS (L1 SRCNN)")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    print(f"\n✅ Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
