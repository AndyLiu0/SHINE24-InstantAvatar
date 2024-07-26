import glob
import os
import cv2
import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class Evaluator(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""
    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    # custom_fwd: turn off mixed precision to avoid numerical instability during evaluation
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):
        # torchmetrics assumes NCHW format
        rgb = rgb.permute(0, 3, 1, 2).clamp(max=1.0)
        rgb_gt = rgb_gt.permute(0, 3, 1, 2)
        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }


@hydra.main(config_path="./confs", config_name="SNARF_NGP_refine")
def main(opt):
    
    base_path = "/root/workspace/InstantAvatar/"
    print("Loading renders")
    imgs = [cv2.imread(fn) for fn in sorted(glob.glob(base_path + "outputs/andy/baseline/andy2/animation/given/*.png"))]
    print("Applying thresholds")
    for idx, img in enumerate(imgs):
        ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY)
        img[thresh == 255] = 0
        write_path = base_path + "outputs/andy/baseline/andy2/animation/compare/" + str(idx).zfill(5) + ".png"
        cv2.imwrite(write_path, img)
    imgs = [cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5) for img in imgs]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    imgs = [torch.tensor(img).cuda().float().unsqueeze_(0) / 255.0 for img in imgs]
    print("Loading ground truths")
    gts = [cv2.imread(fn) for fn in sorted(glob.glob(base_path + "data/custom/andy2/masked_images/*.png"))]
    gts = [cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5) for img in gts]
    for idx, img in enumerate(gts):
        write_path = base_path + "outputs/andy/baseline/andy2/animation/gts/" + str(idx).zfill(5) + ".png"
        cv2.imwrite(write_path, img)
    gts = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in gts]
    gts = [torch.tensor(img).cuda().float().unsqueeze_(0) / 255.0 for img in gts]


    evaluator = Evaluator()
    evaluator = evaluator.cuda()
    evaluator.eval()

    #H, W = imgs[0].shape[:2]
    #W //= 3
    print("Evaluating")
    results = [evaluator(img, gts[idx]) for idx, img in enumerate(imgs)]
    
    with open("results.txt", "w") as f:
        psnr = torch.stack([r['psnr'] for r in results]).mean().item()
        print(f"PSNR: {psnr:.2f}")
        f.write(f"PSNR: {psnr:.2f}\n")

        ssim = torch.stack([r['ssim'] for r in results]).mean().item()
        print(f"SSIM: {ssim:.4f}")
        f.write(f"SSIM: {ssim:.4f}\n")

        lpips = torch.stack([r['lpips'] for r in results]).mean().item()
        print(f"LPIPS: {lpips:.4f}")
        f.write(f"LPIPS: {lpips:.4f}\n")


if __name__ == "__main__":
    main()
