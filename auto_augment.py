from PIL import Image, ImageEnhance, ImageOps
import random
import torch
import torchvision.transforms as T
import numpy as np

class ShearX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
        self.magnitude = 4
    def __call__(self, x):
        return x.transform(
            x.size, Image.AFFINE, (1, self.magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)


class ShearY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
        self.magnitude = 8
    def __call__(self, x):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, self.magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)


class TranslateX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
        self.magnitude = 9
    def __call__(self, x):
        return x.transform(
            x.size, Image.AFFINE, (1, 0,self.magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=self.fillcolor)


class TranslateY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
        self.magnitude = 9

    def __call__(self, x):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, 0, 1, self.magnitude * x.size[1] * random.choice([-1, 1])),
            fillcolor=self.fillcolor)


class Rotate(object):
    def __init__(self):
    # from https://stackoverflow.com/questions/
    # 5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        self.magnitude = 9
    def __call__(self, x):
        rot = x.convert("RGBA").rotate(self.magnitude * random.choice([-1, 1]))
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(x.mode)


class Color(object):
    def __init__(self):
        self.magnitude = 8
    def __call__(self, x):
        return ImageEnhance.Color(x).enhance(1 + self.magnitude * random.choice([-1, 1]))


class Posterize(object):
    def __init__(self):
        self.magnitude = 8
    def __call__(self, x):
        return ImageOps.posterize(x, self.magnitude)


class Solarize(object):
    def __init__(self):
        self.magnitude = 8
    def __call__(self, x):
        return ImageOps.solarize(x, self.magnitude)


class Contrast(object):
    def __init__(self):
        self.magnitude = 8
    def __call__(self, x):
        return ImageEnhance.Contrast(x).enhance(1 + self.magnitude * random.choice([-1, 1]))


class Sharpness(object):
    def __init__(self):
        self.magnitude = 8
    def __call__(self, x):
        return ImageEnhance.Sharpness(x).enhance(1 + self.magnitude * random.choice([-1, 1]))


class Brightness(object):
    def __init__(self):
        self.magnitude = 8
    def __call__(self, x):
        return ImageEnhance.Brightness(x).enhance(1 + self.magnitude * random.choice([-1, 1]))


class AutoContrast(object):
    def __init__(self):
        self.magnitude = 8
    def __call__(self, x):
        return ImageOps.autocontrast(x)


class Equalize(object):
    def __init__(self):
        self.magnitude = 8
    def __call__(self, x):
        return ImageOps.equalize(x)


class Invert(object):
    def __init__(self):
        self.magnitude = 8
    def __call__(self, x):
        return ImageOps.invert(x)


augmentations_list = [ T.RandomApply(ShearX(), p=0.8), 
                    T.RandomApply(ShearY(), p=0.8), 
                    T.RandomApply(TranslateX(), p=0.8), 
                    T.RandomApply(TranslateY(), p=0.8), 
                    T.RandomApply(Rotate(), p=0.8), 
                    T.RandomApply(Color(), p=0.8), 
                    T.RandomApply(Posterize(), p=0.8), 
                    T.RandomApply(Solarize(), p=0.8), 
                    T.RandomApply(Contrast(), p=0.8), 
                    T.RandomApply(Sharpness(), p=0.8),
                    T.RandomApply(Brightness(), p=0.8),
                    T.RandomApply(AutoContrast(), p=0.8),
                    T.RandomApply(Equalize(), p=0.8),
                    T.RandomApply(Invert(), p=0.8)]
# x = torch.randn((3, 224, 224))
# x = T.ToPILImage()(x)

# shear = Sharpness()

# x1 = shear(x)
# x1 = np.asarray(x1)
# print(x1)