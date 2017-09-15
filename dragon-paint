#sample dragon sketch to color
from matplotlib import pyplot as plt
dragon=cv2.imread('Dragon.bmp',0)
plt.imshow(dragon, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

#This is code from DragonPaint, a computer vision/graphics/machine learning program for automating sketch painting for cartoon 
#drawing and animation production, for flat/cel style character drawing/coloring, e.g. The Simpsons. Project is in progress.
#K.G.Greene
#9/15/17

#DRAGONPAINT AUTOPAINTS SKETCHES BY CHARACTER TYPE USING IMAGE TO IMAGE TRANSLATION WITH CONDITIONAL ADVERSARIAL NETS
#The goal of DragonPaint is to automate the painting of sketches of animated characters in a consistent
#way based on one or very few examples or rules/patterns we can state for a given character type, e.g. for dragons: 
#paint body green, spikes, yellow, and leave eye uncolored; for Bart Simpson: paint head and body parts yellow, shirt red,
#pants blue (or whatever clothes specified for a specific episode) and leave eyes uncolored.

#DragonPaint adapts Isola, et. al.'s GAN based machine learning algorithm/code from their paper, Image to Image Translation 
#with Conditional Adversarial Nets, to create a colored sketch/cartoon character from an uncolored one. Based on Isola, 
#et. al.'s applications, the estimated number of matched pairs needed (sketch + same sketch colored in) to train the model 
#is 400-1000.

#DRAGONPAINT MINIMIZES MANUAL WORK IN TRAINING SET CREATION
#A major subgoal of DragonPaint is to require the smallest number possible of manually drawn and manually colored images  
#for the creation of the ML training set.

#A later section will address reducing the number of manually drawn images. This section assumes that a collection of good
#drawn sketches exists and focuses on reducing the number of manually colored paired images we need.

#GOAL=ZERO MANUALLY COLORED TRAINING IMAGES IN TRAINING PAIRS
#The goal is to require only a single manually colored sketch per character type (or zero if a color scheme can be 
#described and used without actually coloring a sketch.)

#CREATING COLORED HALF OF TRAINING PAIR WITHOUT MANUAL PAINTING
#GOOD SKETCH AUTOPAINT
#I. We create a "good sketch" autopainting program using components and geometric relationships so that from any goodSketch 
#we get a goodSketchColored and make (goodSketch, goodSketchColored) a pair for the training set.
#EXPANDING TO BAD SKETCHES USING GOOD SKETCH COLORED PAIR
#II. We expand our training set by modifying a goodSketch and/or by modifying the pair (goodSketch, goodSketchColored).
#A. Given a pair (goodSketch,goodSketchColored) we expand the training set to include bad sketches by e.g. erasing part of 
#goodSketch, pairing it with the original goodSketchColored and adding the pair (goodSketchWithErasures,goodSketchColored) 
#to the training set.
#B. We also expand the training set by modifying both images in the pair in the same way, e.g. by cropping the goodSketch 
#and its colored mate (same crop parameters) so that the background becomes disconnected or smaller than the body and then 
#adding the pair (goodSketchCropped,goodSketchColoredCropped) to the training set.

#GOOD SKETCH DEFINITION
#A "good sketch" for a dragon is based on observations of a few of the simpler dragon sketches (e.g. leave off ears, claws, 
#don't let arms create a disconnected background). A "good sketch" has the following properties:
#1. Black lines on white background.
#2. Lines connect so paint by "paint bucket"/components works.
#3. Background is connected and is bigger than any other white component.
#4. Body is the next biggest white component.
#5. Only interior part (inside the body) is the eye.
#6. Only body parts are body, spikes, eye (without eyelid).

#LABELING GOOD SKETCHES FOR PAINT BY PART
#The code shown here, SketchComponents, is part of I, the "good sketch" autopainting program. SketchComponents takes a 
#"good sketch" and labels the parts, preparing them to be automatically colored with a "paint by body part" mapping. 
  
#CODE SAMPLE
import math, cv2, numpy as np
from skimage import measure 
from skimage.measure import regionprops
from functools import reduce

class SketchComponents:
    #Part of project DragonPaint

    #SketchComponents stores sketch image and uses component information (area, color, distance between components) to 
    #label sketch parts of image so they can be colored according to a part coloring map, e.g. color dragon body green, 
    #spikes yellow and leave eyes white.
    
    #Image = black line cartoon sketch on white background from "Paint" program with well connected lines 
    #and certain geometric relationships (e.g. background bigger than body and body bigger than any other dragon part).
    
    def __init__(self,image,cartoonCharType='flower'):
        #For the two cartoonCharTypes ('flower' and 'dragon') finds white components, background, body and spikes at init
        #For cartoonCharType=='dragon', finds distance between each component and background, finds line width and finds 'eye'
        #Depending on incoming data, might want to change to grayscale or scale to standard size
        
        #store sketch image, threshold to black and white, calculate labels and regionprops for connected components 
        self.image=image
        ret,self.blackWhiteSketch = cv2.threshold(self.image,200,255,cv2.THRESH_BINARY) 
        #for region in regions we can access e.g. region.label, region.area, region.coords
        self.regions = self.labelComponents()   
        
        #label background and body (largest and second largest white components) and spikes+ (other white components)
        self.backgroundLabel, self.bodyLabel, self.spikeLabels = self.setBackgroundBodySpikes()
        
        if cartoonCharType=='dragon':
            self.lineWidth=self.setLineWidth()
            self.eyeLabels, self.spikeLabels = self.setEyeSpikes()

    #BACKGROUND V. BODY V. SPIKES+ for dragons
    #(OR BACKGROUND V. CENTERS V. PETALS for flowers)
    
    #Use components' size and color (white) to distinguish between background, body and spikes+.
    #The largest white component is background. The second largest is body. 
    #The rest of the white components are spikes (or spikes+other for dragon)
    
    #LABEL CONNECTED COMPONENTS
    def labelComponents(self):
        return regionprops(measure.label(self.blackWhiteSketch))
    
    #LABEL BACKGROUND, BODY, SPIKES
    def setBackgroundBodySpikes(self):
        #background is largest white region, body is next largest; spikeLabels is the remaining white components
        #By default, measure.label assigns all black pixels label 0.
        spikeLabels=sorted([region.label for region in self.regions if region.label!=0],key=lambda label: self.regions[label-1].area)
        backgroundLabel=spikeLabels.pop()
        bodyLabel=spikeLabels.pop()
        return backgroundLabel, bodyLabel, spikeLabels
        
    #SPIKES V. EYE
    #For dragon only; not needed for flowers.
    #We can use that the eye is in the interior of the body and the spikes are on the edge to distinguish between them.
    #EucDist(eye,background)>3*lineWidth and EucDist(eye,body)<2*lineWidth
    #Eye may have multiple white components but distance to background > 3*lineWidth holds for all of them.
    #We can get line width from the distance between body and background: lineWidth=EucDist(body,background)
    #Look for a faster way than this for alternate algorithm because the component distance calculation is very slow.

    def calcEucDistCompToBackgrd(self,label):
        #Calculates Euclidean distance between background component and the component with label, 'label'
        return math.sqrt(min((ptLbl[0]-ptBckgrd[0])**2+(ptLbl[1]-ptBckgrd[1])**2 for ptLbl in 
                              self.regions[label-1].coords for ptBckgrd in self.regions[self.backgroundLabel-1].coords))
        
    def setLineWidth(self):
        #Finds distance between body and background to calculate drawing line width.
        return self.calcEucDistCompToBackgrd(self.bodyLabel)
    
    def setEyeSpikes(self):
        #Use the component to background distance requirement to get eyeLabels, then remove those from spikeLabels
        eyeLabels = list(filter(lambda label:self.calcEucDistCompToBackgrd(label)>=2*self.lineWidth,self.spikeLabels))
        spikeLabels = list(filter(lambda label:label not in eyeLabels,self.spikeLabels))
        return eyeLabels, spikeLabels
