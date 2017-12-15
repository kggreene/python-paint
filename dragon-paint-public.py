# An A is a black and white sketch and a B is the colored version
# We'll start with ~30 black and white sketches (A's) and are trying to create
# 400-1000 AB pairs to use for a TensorFlow model
# Class SketchComponents stores a black and white sketch A and colors it by
# geometric rules (which work on 'rule conforming' sketches) to get B.
# Class Augmentations has the functions used to augment (to get from 30 to 400-100).
# Most are used on both A and B with the same parameters to get A' and B' e.g. mirrorFlipPair.
# I put some test code at the bottom that I was using for testing the classes and functions
# You'll want to grab Flower1.png and Dragon4.png to use for testing.


import cv2
import numpy as np
from skimage import measure
from skimage.measure import regionprops
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from sklearn.preprocessing import normalize


class Augmentations():
    # Part of project DragonPaint
    # Augmentations is a collection of augmentation transformations for AB pairs:
    # randxyScaledSkewedPair, randTranslationPair, randRotationPair,
    # mirrorFlipPair, radiusCubedPair (r->r**3),
    # elasticTransformColorSetPair (Gaussian filters)

    # still need elasticDeformationGray (done for one img but need set version?
    # incorporate read BW, transform, color into the function?)
    # still need erasures, crops
    # still need (here or elsewhere) the sequences of transformations

    # AFFINE TRANSFORMATIONS:
    def randxyScaledSkewedPair(self, imgA, imgB):
        # small x and y scale and skew
        # fills in background white, might crop image

        xscale = np.random.uniform(0.8, 1)
        yscale = np.random.uniform(0.8, 1)
        skewFactor = np.random.uniform(0, 0.1)
        M = np.float32([[xscale, 0, 0], [skewFactor, yscale, 0]])
        return self.warpAffinePair(imgA, imgB, M)        

    def randTranslationPair(self, imgA, imgB):
        # small translation in x and y
        # fills in background white, might crop image

        transMax = 20
        transX = np.random.randint(transMax)
        transY = np.random.randint(transMax)
        M = np.float32([[1, 0, transX], [0, 1, transY]])
        return self.warpAffinePair(imgA, imgB, M)        

    def randRotationPair(self, imgA, imgB, characterType='flower'):
        # chooses random rotation between 0 and maxRotation
        # fills in background white, might crop image

        maxRotation = {'flower': 360, 'dragon': 10}
        rotation = np.random.randint(maxRotation[characterType])
        # note: cols and rows are undefined here
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
        return self.warpAffinePair(imgA, imgB, M)

    def warpAffinePair(self, imgA, imgB, M):
        # please rename this function if needed        
        size = len(imgA)    
        return [cv2.warpAffine(img,
                              M,
                              (size, size),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
                for img in (imgA, imgB)]
        
    def mirrorFlipPair(self, imgA, imgB):
        # flips across y axis

        flipA = cv2.flip(imgA, 1)
        flipB = cv2.flip(imgB, 1)
        return flipA, flipB

    # ELASTIC DEFORMATIONS/GAUSSIAN BLUR:
    def elasticTransformColorSetPair(self, imgA, imgB):
        # Assumes both imgA and imgB are 3D arrays (i.e. color not grayscale)
        # Creates a list of four transformed AB pairs with different parameters
        # for each pair

        alphas = [50, 200, 800, 1500]
        sigmas = [3, 4, 7, 8]
        Alist = []
        Blist = []
        for alpha, sigma in zip(alphas, sigmas):
            transA, transB = self.elasticTransformColorPair(imgA,
                                                            imgB,
                                                            alpha,
                                                            sigma)
            Alist.append(transA)
            Blist.append(transB)
        return Alist, Blist

    def elasticTransformColorPair(self,
                                  imgA,
                                  imgB,
                                  alpha,
                                  sigma,
                                  random_state=None):
        # function for elasticTransformColorSetPair
        # elastic deformations/random displacement fields
        # modified from Github https://gist.github.com/erniejunior/601cdf56d2b424757de5;
        # based on Simard, et al
        # Assumes both are 3D arrays (i.e. color not grayscale)
        # Returns a single transformed AB pair

        if random_state is None:
            random_state = np.random.RandomState(None)
        shape = imgA.shape
        dx, dy = [self.rename_this_function(shape, alpha, sigma) for _ in range(2)]
        dz = np.zeros_like(dx)
        
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = [np.reshape(array, (-1, 1)) for array in (y+dy, x+dx, z)]
        
        return [map_coordinates(img,
                                indices,
                                order=1,
                                mode='reflect').reshape(img.shape)
                for img in (imgA, imgB)]


    def _rename_this_function(self, shape, alpha, sigma):
        # what does this function mean? 
        
        return gaussian_filter((random_state.rand(*shape) * 2 - 1),
                                sigma,
                                mode="constant",
                                cval=0) * alpha
    
    def elasticDeformationGray(self, img, alpha, blurSize):
        # requires grayscale image
        # so unlike the AB pair transformations, read in as grayscale,
        # transform and change to BGR, then color A' to get B'
        # function for elasticDeformationGrayPairSet
        # similar to elasticTransformColor but different parameters and results
        # turning out enough different worth using good parameters:
        # alphas=[100,150,150,150,125]
        # blurSizes=[99,99,125,155,155]

        shape = img.shape

        dx = calculateRandomDeformation(shape, alpha, blurSize)
        dy = calculateRandomDeformation(shape, alpha, blurSize)
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        distorted_image = map_coordinates(img,
                                          indices,
                                          order=1,
                                          mode='constant',
                                          cval=255)

        return distorted_image.reshape(img.shape)


    def calculateRandomDeformation(shape, alpha, blurSize):
        # there might be a better name!
        field = np.zeros((shape[0], shape[0]))
        for i in range(shape[0]):
            for j in range(shape[0]):
                field[i][j] = np.random.uniform(-1, 1)
        blur = cv2.GaussianBlur(field, (blurSize, blurSize), 0)
        normBlurField = normalize(blur)
        field = alpha * normBlurField
        return field

        
    # RADIUS CUBED/DISC HOMEOMORPHISM
    def radiusCubedPair(self, imgA, imgB):
        # for flower, not for dragon
        # homeomorphism of unit disk shrinks center disproportionally;
        # based on r->r**3

        scale = np.random.uniform(0.4, 0.6)
        return self.radiusCubed(imgA, scale), self.radiusCubed(imgB, scale)

    def cubeRoot(self, x):
        # function for radiusCubed

        if x >= 0:
            return x ** (1 / 3)
        else:
            return -(abs(x) ** (1 / 3))

    def radiusCubed(self, imgA, scale=.5):
        # function for radiusCubedPair
        # translate and scale to fit in unit disk; r, theta ->r**3, theta;
        # rescale, translate to original square

        rows, cols, ch = imgA.shape
        newA = np.zeros((rows, cols, ch))
        mid = rows // 2
        size = rows
        scaleFactor = size * scale
        for row in range(size):
            for col in range(size):
                rowOld = row
                colOld = col

                # move origin to middle
                rowOld = row - mid
                colOld = col - mid

                # scale to fit square in unit disc
                rowOld = rowOld / scaleFactor
                colOld = colOld / scaleFactor

                # transform unit disc in a way that shrinks center more than edge
                # r->r**3
                rSquared = self.cubeRoot(rowOld ** 2 + colOld ** 2)
                if rSquared:
                    rowOld = rowOld / rSquared
                    colOld = colOld / rSquared

                # scale back out of unit disc to full size disc/square
                rowOld = rowOld * scaleFactor
                colOld = colOld * scaleFactor
                # move origin from middle to upper left
                rowOld = rowOld + mid
                colOld = colOld + mid

                # modify so rowOld and colOld are valid indices
                rowOld = int(rowOld)
                colOld = int(colOld)
                rowOld = max(0, rowOld)
                rowOld = min(size - 1, rowOld)
                colOld = max(0, colOld)
                colOld = min(size-1, colOld)
                newA[row][col] = imgA[rowOld][colOld]
        return newA


class SketchComponents():
    # Part of project DragonPaint
    # SketchComponents stores sketch image and uses component information
    # (area, color, bounding box) to label sketch parts of image so they can
    # be colored according to a part coloring map, e.g. color dragon body
    # green, spikes yellow and leave eyes white.
    # Image = black line cartoon sketch on white background from "Paint"
    # program with well connected lines and certain geometric relationships
    # (e.g. background bigger than body and body bigger than any other dragon
    # part).

    def __init__(self, image, cartoonCharType='flower'):
        # For the two cartoonCharTypes ('flower' and 'dragon') finds white
        # components, background, body and spikes at init
        # For cartoonCharType=='dragon', finds distance between each component
        # and background, finds line width and finds 'eye'
        # Depending on incoming data, might want to change to grayscale or
        # scale to standard size

        # store sketch image, threshold to black and white, calculate labels
        # and regionprops for connected components
        self.image = image
        ret, self.blackWhiteSketch = cv2.threshold(self.image,
                                                   200,
                                                   255,
                                                   cv2.THRESH_BINARY)

        # for region in regions we can access e.g. region.label, region.area, region.coords
        self.regions = self.labelComponents()

        # label background and body (largest and second largest white
        # components) and spikes+ (other white components)
        self.backgroundLabel, self.bodyLabel, self.spikeLabels = self.setBackgroundBodySpikes()

        if cartoonCharType == 'dragon':
            self.eyeLabels, self.spikeLabels = self.setEyeSpikes()

    # BACKGROUND V. BODY V. SPIKES+ for dragons
    # (OR BACKGROUND V. CENTERS V. PETALS for flowers)

    # Use components' size and color (white) to distinguish between background,
    # body and spikes+.
    # The largest white component is background. The second largest is body.
    # The rest of the white components are spikes (or spikes+other for dragon)

    # LABEL CONNECTED COMPONENTS
    def labelComponents(self):
        return regionprops(measure.label(self.blackWhiteSketch))

    # LABEL BACKGROUND, BODY, SPIKES
    def setBackgroundBodySpikes(self):
        # background is largest white region, body is next largest; spikeLabels
        # is the remaining white components
        # By default, measure.label assigns all black pixels label 0.

        spikeLabels = sorted([region.label for region in self.regions
                              if region.label != 0],
                             key=lambda label: self.regions[label - 1].area)
        backgroundLabel = spikeLabels.pop()
        bodyLabel = spikeLabels.pop()
        return backgroundLabel, bodyLabel, spikeLabels

    # SPIKES V. EYE
    def setEyeSpikes(self):
        # For dragon only; not needed for flowers.
        # The eye is in the interior of the body and the spikes are on the edge.
        # If "spike"'s puffed up bounding box intersects the background it's a
        # spike; if not, it's an eye (or part of an eye)

        linePlus = 5
        # linePlus gives a puffed up bounding box to account for drawing line
        # thickness; it may need to be adjusted if pictures have been scaled
        # down from original (400x400 image with 5 pixel line.)

        eyeLabels = []
        backgrd = self.regions[self.backgroundLabel-1].coords
        for spikeLabel in self.spikeLabels:
            intersection = False
            rowmin, colmin, color0, rowmax, colmax, color2 = self.regions[spikeLabel - 1].bbox
            for point in backgrd:
                if (not intersection) and point[0] > rowmin - linePlus and point[0] < rowmax + linePlus:
                    if point[1] > colmin - linePlus and point[1] < colmax + linePlus:
                        intersection = True
            if not intersection:
                eyeLabels.append(spikeLabel)
        newSpikeLabels = [label for label in self.spikeLabels
                          if label not in eyeLabels]
        return eyeLabels, newSpikeLabels


# TEST CODE FOR SKETCHCOMPONENTS
# run line with "Flower1.png" to see flower or the line with "Dragon4.png"
# to see dragon
flower = SketchComponents(cv2.imread("Flower1.png"))

# flower=SketchComponents(cv2.imread("Dragon4.png"), 'dragon')
imgColored = flower.image

# color spikes yellow
for spikeLabel in flower.spikeLabels:
    spike = flower.regions[spikeLabel - 1].coords
    for point in spike:
        imgColored[point[0], point[1], 0] = 0
        imgColored[point[0], point[1], 1] = 241
        imgColored[point[0], point[1], 2] = 253

# color center orange
body = flower.regions[flower.bodyLabel - 1].coords
for point in body:
    imgColored[point[0], point[1], 0] = 0
    imgColored[point[0], point[1], 1] = 159
    imgColored[point[0], point[1], 2] = 255
cv2.imshow('image colored', imgColored)
cv2.waitKey(0)
cv2.destroyAllWindows()


# TEST CODE FOR AUGMENTATIONS (one function)
flowerGray = cv2.imread('Flower1.png', 0)
print(flowerGray.shape)
augment = Augmentations()
elasticGray = augment.elasticDeformationGray(flowerGray, 100, 99)
cv2.imshow('elasticGray', elasticGray)
cv2.waitKey(0)
cv2.destroyAllWindows()
