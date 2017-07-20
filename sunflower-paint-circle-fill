# Use connected components to find and color the interior of a circle

# For a black circle (not filled in) on a white background, 
# the biggest white component is background, second biggest is interior of circle or "body"

import numpy as np
from skimage.measure import label
from skimage.measure import regionprops

# Create a black image and turn it white and draw a black circle
img = np.zeros((600,600), np.uint8)
img[:] = 255
cv2.circle(img,(300,300),100,0,5)

#find connected components
labels=label(img,background=-1)
props=regionprops(labels)

# find two biggest white connected components
#find color of points in region with label i+1
regionColor=[]
i=0
for i in range(0,len(props)):
    row,column=props[i].coords[0][0],props[i].coords[0][1]
    regionColor.append(img[row][column])

#make a list of sizes of white components
whiteSize=[]
for i in range(0,len(props)):
    if regionColor[props[i].label-1]==255:
        whiteSize.append(props[i].area)
sizeSorted=sorted(whiteSize,reverse=True)
biggestSize=sizeSorted[0]
secondBiggestSize=sizeSorted[1]

for region in props:
    if region.area==biggestSize:
        backgroundLabel=region.label
    if region.area==secondBiggestSize:
        if region.label!=backgroundLabel:
            bodyLabel=region.label

#change color of body to gray
body=props[bodyLabel-1].coords
for point in body:
    img[point[0]][point[1]]=100

#display image of circle filled with gray
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
