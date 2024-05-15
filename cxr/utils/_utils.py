import torch
import os
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x



first_split = "FINDINGS AND IMPRESSION:"
second_split = "FINDINGS:"
third_split = "IMPRESSION:"
fourth_split = "FINDINGS"

def get_before_findings(t):
    """We want to get the earliest beginning of a findings paragraph"""
    if first_split in t:
        return t.split(first_split)[0] + first_split
    if second_split in t:
        return t.split(second_split)[0] + second_split
    if third_split in t:
        return t.split(third_split)[0] + third_split
    return t.split(fourth_split)[0] + fourth_split


FINDINGS_STARTS = [first_split, second_split, third_split, fourth_split]
FINAL_REPORT_START = " " * 33 + "FINAL REPORT"

