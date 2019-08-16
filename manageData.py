from shutil import copyfile
import os

def main():
    for dir in os.listdir("./dicom-images-test"):
        for dir2 in os.listdir("./dicom-images-test/"+dir):
            for file in os.listdir("./dicom-images-test/"+dir+"/"+dir2):
                #print(file)
                copyfile("./dicom-images-test/"+dir+"/"+dir2+"/"+file,"./test_images/"+file)



if __name__ == "__main__":
  main()
