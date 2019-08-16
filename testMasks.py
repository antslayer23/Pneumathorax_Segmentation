import numpy as np
import csv
import pandas as pd
import cv2 as cv
import pydicom as dicom
import os
#from shutil import copyfile
import numpy
import re
from PIL import Image
from mask_functions import rle2mask
from mask_functions import mask2rle
def main():
    test = 4
    encodingsV = pd.read_csv("train-rle.csv")
    encodings = encodingsV.values
    imageID = encodings[test,0]
    #print(imageID)
    imageFront = re.search("\d+\.\d+\.\d+\.\d+\.\d+\.\d+\.\d+\.", imageID).group(0)
    imageBack = re.split("\d+\.\d+\.\d+\.\d+\.\d+\.\d+\.\d+\.\d+", imageID,1)[1]
    imageTail = re.split("\d+\.\d+\.\d+\.", imageBack,1)[1]
    imageTail = int(imageTail)
    imageTail2 = imageTail-1
    imageTail3 = imageTail-2
    imageBack = re.search("\d+\.\d+\.\d+\.", imageBack).group(0)

    #print(imageFront)
    #print(imageTail)
    string = './dicom-images-train/'+imageFront+"2."+imageBack+str(imageTail2)+'/'+imageFront+"3."+imageBack+str(imageTail3)+'/'+imageFront+"4."+imageBack+str(imageTail)+'.dcm'
    #print(string)
    img = dicom.read_file(string)
    print(img.pixel_array.shape)


    maskString = encodings[test,1]

    mask = rle2mask(maskString,img.pixel_array.shape[0],img.pixel_array.shape[1])

    img2 = Image.fromarray(mask).convert("RGBA")

    im = Image.fromarray(img.pixel_array).convert("RGBA")



    im.show()
    img2.show()

    alphaComposited = Image.blend(im, img2, 0.5)
    alphaComposited.show()


if __name__ == "__main__":
  main()
