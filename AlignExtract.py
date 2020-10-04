import cv2 as cv
import numpy as np
from skimage.transform import rotate
import math



def extractRegion(im7D,ang,shift,ysizeT,xsizeT,ysizeOut=None,xsizeOut=None):
    """ This function extracts a region from a 7D image.
        You tell it the size, location and rotation of the region to extract.
        I.e. it is made to work with findSelection() below.
        im7D must be in our starndard form, i.e. with dimensions:
        (times,regions,montages,zslices,channels,ysize,xsize)
        You just pass it the angle and shift from findYourSection().
        You can pad the output to make it a certain size if you want, using 
        ysizeOut and xsizeOut.
        If you give it an image with 5D it assumes you want to add time and 
        field dimensions. I.e. it assumes you have done something likes 
        extractRegion(im7D[t,f]) and lost the first 2 dimensions.
        
        How it works:
        to extract template from image given ang and shift is easy:
        rotate point [0,0] in template by ang
        (remember how this requires ysizeT,xsizeT b/c it takes resizing of 
        the template during rotation into account)
        (and it was this resized rotated template that was matched to an 
        image region)
        then shift that rotated [0,0] point (it will now match the top left 
        corner of the template but in the image coordinates)
        then rotate both the image and point by -ang (now taking into 
        account the resizing of the image)
        (now the template will be sat squarely in the rotated image)
        now just add ysizeT,xsizeT to the point to find the limits
    """
    # package template shape to stop code bloating
    shapeT = (ysizeT,xsizeT)
    
    # add axes if needed
    if len(im7D.shape)==5:
        im7D = im7D[np.newaxis,np.newaxis]
    if len(im7D.shape)==6:
        im7D = im7D[np.newaxis]
    
    times,regions,montages,zslices,channels,ysizeI,xsizeI = im7D.shape
    # package these to stop code bloating
    shapeI = (ysizeI,xsizeI)
    
    # the top left corner of the template, in the template coord system:
    corner = [0,0]
    
    # now rotated as template rotates and shift added 
    # so it is aligned into the image coordinate system:
    corner = [sum(x) for x in zip(rotCoord_NoCrop(corner,*shapeT,ang),shift)]
    
    # and now rotated as image rotates so it is square:
    corner = [int(pos) for pos in rotCoord_NoCrop(corner,*shapeI,-ang)]
    
    # now arrange the im7D so it can apply all rotations at once:
    prod5D = times*regions*montages*zslices*channels
    im7D = np.swapaxes(np.swapaxes(np.reshape(im7D,(prod5D,*shapeI)),0,1),1,2)

    # rotate the image so template can be extracted as a square
    im7D = rotate_image(im7D,-ang)
    
    # do extraction:
    im7D = im7D[corner[0]:corner[0] + ysizeT,corner[1]:corner[1] + xsizeT]
    
    # put it back in the original shape:
    # package first 5 dimensions to stop code bloating
    shape5D = (times,regions,montages,zslices,channels)
    im7D = np.swapaxes(np.swapaxes(im7D,1,2),0,1).reshape(*shape5D,*shapeT)
    
    if ysizeOut:
        pY = ysizeOut - ysizeT
        pad = ((0,0),(0,0),(0,0),(0,0),(0,0),(pY//2,math.ceil(pY/2)),(0,0))
        im7D = np.pad(im7D,pad)
    if xsizeOut:
        pX = xsizeOut - xsizeT
        pad = ((0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(pX//2,math.ceil(pX/2)))
        im7D = np.pad(im7D,pad)
    
    # now return the extraction:
    return im7D



'''
# this function extracts the template from a tif of our starndard form (times,regions,montages,zslices,channels,ysize,xsize)
# you just pass it the angle and shift from findYourSection()
# to extract template from image given ang and shift is easy:
# rotate point [0,0] in template by ang
# (remember how this requires ysizeT,xsizeT b/c it takes resizing of the template during rotation into account)
# (and it was this resized rotated templated that was matched to an image region)
# then shift that rotated [0,0] point (it will now match the top left corner of the template but in the image coordinates)
# then rotate both the image and point by -ang (now taking into account the resizing of the image)
# (now the template will be sat squarely in the rotated image)
# now just add ysizeT,xsizeT to the point to find the limits
def extractTemplate(image,ang,shift,ysizeT,xsizeT):
    
    ysizeI,xsizeI = image.shape
    
    # the top left corner of the template, in the template coord system:
    corner = [0,0]
    
    # now rotated as template rotates and shift added, so it is aligned into the image coordinate system:
    corner = [sum(x) for x in zip(rotateImageCoords(corner,ang,ysizeT,xsizeT,convertToRadians=True),shift)]
    
    # and now rotated as image rotates so it is square:
    corner = [int(pos) for pos in rotateImageCoords(corner,-ang,ysizeI,xsizeI,convertToRadians=True)]
    
    # rotate the image so template can be extracted as a square
    imageR = rotate(image,-theAngle,resize=True)
    
    # now return the extraction:
    return imageR[corner[0]:corner[0] + ysizeT,corner[1]:corner[1] + xsizeT]

'''



# this does the template matching by masked normalised cross-correlation
# you give it the image and the template and it returns a tuple (angle,[yshift,xshift])
# positive angle is anti-clockwise 
# [yshift,xshift] is the shift of the top-left corner of the rotated template from the top left corner of the image
# therefore !watch out! you have to rotate any template coordinates before applying shift if you want to find them in image
# it does a cross-correlation of the template with the image, the template is masked where there is black due to... 
# ...the image resize after rotation, this stops the image in these areas entering the nomralisation, which would favour...
# ... small rotations since big ones have large black areas
# the angles search is what really takes time, the user can define a limit maxAngleD in degrees
# it is also limited to rotations of template which actually fit in the image
def findRegion(image,template,anglePrecision,maxAngleD):
    
    # get sizes, they will be needed
    ysizeI,xsizeI = image.shape
    ysizeT,xsizeT = template.shape
    
    # checking all angles take time so we only go to +-maxAngleD supplied by user
    # but here we might reduce it more... if you can't rotate the template inside the image... 
    # ...without the template going outside the image then we stop searching angles when the rotated template...
    # ...reaches this limit. That's because things get complicated with the normalisation anyway at that point...
    # ...so it's as good a place to stop as any
    maxAngleD = findMaxAngleD(ysizeT,xsizeT,ysizeI,xsizeI,maxAngleD)

    maxValues = []
    maxPos = []

    iRange = range(-int(maxAngleD//anglePrecision)+1,int(maxAngleD//anglePrecision),1)

    for i in iRange:      
        # rotate the template:
        templateR = rotate_image(template,i*anglePrecision)
        # make a mask which will exclude areas of black produced by the rotation
        theMask = np.zeros(templateR.shape)
        theMask[templateR!=0] = True
        theMask = theMask.astype('float32')
        # do the cross-correlation, 10 times faster in cv!
        resultArray = cv.matchTemplate(image,templateR,3,mask=theMask)
        # find the maximum cross-correlation:
        maxValues.append(np.amax(resultArray))
        # find the position of the maximum:
        maxPos.append(np.unravel_index(np.argmax(resultArray), resultArray.shape))
        
    # where in the list is the max of all max values
    maxMaxPos = maxValues.index(max(maxValues))
    
    # this is the angle where the best cross-correlation was found:
    theAngle = iRange[maxMaxPos]*anglePrecision
    
    # this is the shift between the top left corners of the rotated template and image where the best cross-correlation was:
    yShift,xShift = maxPos[maxMaxPos]
    
    return (theAngle,[yShift,xShift])



# this gives the maximum angle in degrees that you can rotate imageS within imageI before it hits an edge
# if the imageS can be rotated completely within imageI then it returns maxAngle
# maxAngle should be in degrees
def findMaxAngleD(ysizeS,xsizeS,ysizeI,xsizeI,maxAngle):
    # the length of the hypotenuse of the section
    hyp = math.sqrt(ysizeS**2 + xsizeS**2)
    
    # if the hypotenuse can fit then the section can fully rotate
    if hyp <= ysizeI:
        YmaxRotD = maxAngle
    # else do the trig to find the angles
    else:
        betaYR = math.atan(ysizeS/xsizeS)
        YmaxRotR = math.asin(ysizeI/hyp) - betaYR 
        YmaxRotD = 180*YmaxRotR/math.pi
    if hyp <= xsizeI:
        XmaxRotD = maxAngle
    else:        
        betaXR = math.atan(xsizeS/ysizeS)
        XmaxRotR = math.asin(xsizeI/hyp) - betaXR
        XmaxRotD = 180*XmaxRotR/math.pi
        
    return min([XmaxRotD,YmaxRotD,maxAngle])



# this is an open cv way to rotated image and resize the image so there's no cropping
# much faster than skimage, end result might be a pixel bigger so it's not exactly the same
# cv.warpAffine can be applied to many 'channels' at once when the array shape is (y,x,channels)
# this is much faster than looping over all channels in python but:...
#...since cv::Mat max allowed no. of channels is 512, it processes in batches of size 512
def rotate_image(mat, angle):
    # angle in degrees

    height, width = mat.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    
    if mat.ndim == 2:
        rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    else:
        rotated_mat = [cv.warpAffine(mat[:,:,i*512:(i+1)*512], rotation_mat, (bound_w, bound_h)) for i in range(mat.shape[2]//512)]
        rotated_mat.append(cv.warpAffine(mat[:,:,(mat.shape[2]//512)*512:], rotation_mat, (bound_w, bound_h)))
        rotated_mat = np.concatenate(rotated_mat,axis=2)
                           
    return rotated_mat



'''

# takes image coordinate and finds those points in the image rotated by ang
# i.e. it is working with the top left corner being [0,0] and y-direction pointing down
# ang should be in radians but set convertToRadians = True if you give it in degrees
# positive ang is anti-clockwise
# you have to give the image ysize,xsize to calculate the shift from image resizing 
# this all assumes you are resizing the image when rotating
# it will only work for angles -90 < ang < 90
def rotateImageCoords(coords,ang,ysize,xsize,convertToRadians=False):
    if convertToRadians:
        ang = ang*math.pi/180
    
    # remember y is flipped in image coords compared with traditional x-y coords, so this looks different to normal!
    coords = [coords[0]*math.cos(ang) - coords[1]*math.sin(ang), coords[1]*math.cos(ang) + coords[0]*math.sin(ang)]
    if ang >= 0:
        coords[0] = coords[0] + xsize*math.sin(ang)
    else:
        coords[1] = coords[1] + ysize*math.sin(-ang)
        
    return coords



def rotateImageCoordsC(coords,rotation_mat):
    
    coordC = [coord[1],coord[0],1]

    np.matmul(rotation_mat,coordC).roun().astype('int')
    
    return [coordC[1],coordC[0]]
    


def getImageRotationMat_NoCrop(ysize,xsize,angle):

    image_center = (xsize/2, ysize/2)
    
    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)
    
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])
    
    bound_x = int(ysize * abs_sin + xsize * abs_cos)
    bound_y = int(ysize * abs_cos + xsize * abs_sin)
    
    rotation_mat[0, 2] += bound_x/2 - image_center[0]
    rotation_mat[1, 2] += bound_y/2 - image_center[1]
    
    return rotation_mat
'''


# this rotates points in exactly the same way as rotate_image rotates an image
def rotCoord_NoCrop(coord,ysize,xsize,angle):

    image_center = (xsize/2, ysize/2)
    
    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)
    
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])
    
    bound_x = int(ysize * abs_sin + xsize * abs_cos)
    bound_y = int(ysize * abs_cos + xsize * abs_sin)
    
    rotation_mat[0, 2] += bound_x/2 - image_center[0]
    rotation_mat[1, 2] += bound_y/2 - image_center[1]

    coordC = [coord[1],coord[0],1]

    coordC  = np.matmul(rotation_mat,coordC).round().astype('int')
    
    return [coordC[1],coordC[0]]