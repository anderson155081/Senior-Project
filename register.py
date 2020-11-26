import os, cv2
from time import sleep

web_cam = cv2.VideoCapture(0)
name = input("enter your name: ")
count =0

path = "dataset/train/"+name

if not os.path.isdir(path):
    os.mkdir(path)

print("get ready and look at the camera....")
sleep(2)
while count<30:
    (ret, im) = web_cam.read()

    count+=1
    cv2.imwrite('dataset/train/%s/%d.png'% (name,count), im)
