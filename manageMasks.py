import os
import pandas as pd
import json
import pydicom as dicom
from mask_functions import rle2mask
from mask_functions import mask2rle
import re
import numpy as np
from PIL import Image

def main():

    encodingsV = pd.read_csv("train-rle.csv")
    encodings = encodingsV.values
    dic = {}
    zeros = 0
    ones = 0
    for i in range(0, encodings.shape[0]):
        if encodings[i,0] not in dic:
            if(str(encodings[i,1]) == " -1"):
                dic[encodings[i,0]] = 0
                zeros = zeros+1
            else:
                dic[encodings[i,0]] = 1
                ones = ones+1

    #print(dic)


    test = 4
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



    with open('maskBinary.json', 'w') as fp:
        json.dump(dic, fp)

    for i in range(0, encodings.shape[0]):
        maskString = encodings[i,1]
        ID = encodings[i,0]
        mask = np.zeros(img.pixel_array.shape)
        if(str(maskString) != " -1"):
            mask = rle2mask(maskString,img.pixel_array.shape[0],img.pixel_array.shape[1])

        img2 = Image.fromarray(mask).convert('L')
        img2.save('./train_masks/'+str(ID)+'.bmp')
        #img2.show()


if __name__ == "__main__":
  main()
