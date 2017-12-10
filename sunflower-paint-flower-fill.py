# Read in daisy flower "Paint" sketch ("flower.png" is in this repository)
# Use connected components to color center one color, petals another
# Find white components, sort by size. Biggest is background, second biggest is flower center, others are petals
# Make a gray daisy and a purple daisy

import cv2, numpy as np
from skimage.measure import label
from skimage.measure import regionprops

#read in image as grayscale
img=cv2.imread('flower.png',0)

#apply binary threshold
ret,thresh1 = cv2.threshold(img,125,255,cv2.THRESH_BINARY)

#find components
labels=label(thresh1,background=-1) 
props=regionprops(labels)

#make list of labels of white regions
whiteLabels=[]
for i in range(0,len(props)):
    row,column=props[i].coords[0][0],props[i].coords[0][1]
    if(thresh1[row][column]==255):
        whiteLabels.append(props[i].label)

#make a list of sizes of white components
whiteSize=[]
for l in whiteLabels:
    whiteSize.append(props[l-1].area)
print(whiteSize)
sizeSorted=sorted(whiteSize,reverse=True)
print(sizeSorted)
biggestSize=sizeSorted[0]
secondBiggestSize=sizeSorted[1]

#find labels for two biggest, background and body
for l in whiteLabels:
    if props[l-1].area==biggestSize:
        backgroundLabel=props[l-1].label
    if props[l-1].area==secondBiggestSize:
        if props[l-1].label!=backgroundLabel:
            bodyLabel=props[l-1].label

#make two toned gray daisy
#color body gray
imgGray=cv2.imread('flower.png',0)
body=props[bodyLabel-1].coords
for point in body:
    imgGray[point[0]][point[1]]=100
    
#color petals light gray
for l in whiteLabels:
    if l!=backgroundLabel and l!=bodyLabel:
        petal=props[l-1].coords
        for point in petal:
            imgGray[point[0]][point[1]]=200

#display gray daisy, hit ESC to close window
cv2.imshow('image gray',imgGray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#make a purple daisy
#imread creates grayscale with a 0, BGR color with a 1
imgPurple=cv2.imread('flower.png',1)

#color body gray purple
body=props[bodyLabel-1].coords
for point in body:
    imgPurple[point[0],point[1],0]=150
    imgPurple[point[0],point[1],1]=50
    imgPurple[point[0],point[1],2]=150
    
#color petals lighter gray purple
for l in whiteLabels:
    if l!=backgroundLabel and l!=bodyLabel: 
        petal=props[l-1].coords
        for point in petal:
            imgPurple[point[0],point[1],0]=255
            imgPurple[point[0],point[1],1]=150
            imgPurple[point[0],point[1],2]=255

#display purple daisy
cv2.imshow('image purple',imgPurple)
cv2.waitKey(0)
cv2.destroyAllWindows()
