import sys
import os
from PIL import Image

def ToGrayscale(pathin, pathout):
    files = filter(lambda f: len(f) > 4 and f[-4:] == ".jpg", os.listdir(pathin))
    for file in files:
        gray_img = Image.open(os.path.join(pathin,file)).convert('L')
        gray_img.save(os.path.join(pathout, 'g_' + file))