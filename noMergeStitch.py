import numpy as np
import math



def noMergeStitch(images,centreList,sizes,xMon=0):
    """ This is the main function that does everything.
         images is a flat list of unmontaged images, 
         i.e. dimensions: (montages,ysize,xsize)
         centreList is the list of centres as given by findCentres().
         sizes defines the size of the returned image. 
         The top and left are cropped or padded with zeros to make 
         it this size.
         This version of the function assembles the image in the most 
         stupid way: it just places each image fully in it's place so 
         any overlapping bits delete the previous image. This means ugly
         edge sections of images are included.
    """
    
    ysize,xsize = np.shape(images[0])
    
    # unpack sizes
    ysize_MFinal = sizes[0]
    xsize_MFinal = sizes[1]
    
    # make numpy array from centreList b/c lists are annoying to manipulate
    centres = np.array(centreList)
    
    # finding span of image centres:
    minCeny, minCenx = np.amin(centres,axis=0)
    maxCeny, maxCenx = np.amax(centres,axis=0)
    
    # initiate final image array, of correct size 
    # (calculated from span of centres)
    ysize_m = maxCeny - minCeny + ysize
    xsize_m = maxCenx - minCenx + xsize
    mergedIm = np.zeros((ysize_m,xsize_m))
    
    # realign centres so they are indices of mergerIm
    minIndexY = minIndexFromCentre(minCeny,ysize)
    minIndexX = minIndexFromCentre(minCenx,xsize)
    centres = centres - np.array([minIndexY,minIndexX])
    
    # first make limits of the images, 
    #i.e. where the image ends and the merging begins
    #!!! for now we just use whole image slices
    limits = []
    for cen in centres:
        limits.append(imageRange_M(cen,ysize,xsize))
    
    #!!! TODO: check corners of limits of diagonal neighbours don't overlap
    # don't have to check horizontal or vertical neighbours, 
    # by definition their limits don't overlap
    
    # first put images in up to limits:
    for i,lim in enumerate(limits):
        mergedIm[lim] = images[i].copy()
    
    # now crop or pad to output size
    if ysize_MFinal > ysize_m:
        mergedIm = np.pad(mergedIm,((ysize_MFinal - ysize_m,0),(0,0)))
    elif ysize_MFinal < ysize_m:
        mergedIm = mergedIm[ysize_m-ysize_MFinal:].copy()
    if xsize_MFinal > xsize_m:
        mergedIm = np.pad(mergedIm,((0,0),(xsize_MFinal - xsize_m,0)))
    elif xsize_MFinal < xsize_m:
        mergedIm = mergedIm[:,xsize_m-xsize_MFinal:].copy()
    
    return mergedIm



def noMergeStitch2(images,centreList,sizes,xMon):
    """ This is the main function that does everything.
         images is a flat list of unmontaged images, 
         i.e. dimensions: (montages,ysize,xsize)
         centreList is the list of centres as given by findCentres().
         sizes defines the size of the returned image. 
         The top and left are cropped or padded with zeros to make 
         it this size.
         xMon is the number of tiles along x.
         This version of the function does it in the 2nd most stupid way...
         Images are partially cropped so overlapping sections generally 
         consist of half one image and half the next... 
         Its easy to allow blank sections if you don't think carefully 
         about which edges to crop and which to allow to be overlaid upon...
         We do it in a bit of a lazy but satifactory way here.
    """
    
    ysize,xsize = np.shape(images[0])
    
    # unpack sizes
    ysize_MFinal = sizes[0]
    xsize_MFinal = sizes[1]
    
    # make numpy array from centreList b/c lists are annoying to manipulate
    centres = np.array(centreList)
    
    # finding span of image centres:
    minCeny, minCenx = np.amin(centres,axis=0)
    maxCeny, maxCenx = np.amax(centres,axis=0)
    
    # initiate final image array, of correct size 
    # (calculated from span of centres)
    ysize_m = maxCeny - minCeny + ysize
    xsize_m = maxCenx - minCenx + xsize
    mergedIm = np.zeros((ysize_m,xsize_m))
    
    # realign centres so they are indices of mergerIm
    minIndexY = minIndexFromCentre(minCeny,ysize)
    minIndexX = minIndexFromCentre(minCenx,xsize)
    centres = centres - np.array([minIndexY,minIndexX])
    
    # first make limits of the images, 
    #i.e. where the image ends and the merging begins
    #!!! for now we just use whole image slices
    limits = []
    limits2 = [[[0,ysize,1],[0,xsize,1]] for i in centres]
    for c,cen in enumerate(centres):
        # first tile goes completely in
        if c == 0:
            limits.append(imageRange_M(cen,ysize,xsize))
        # top row tiles have just left side cropped
        elif c < xMon:
            LXMax = imageRange_M(centres[c-1],ysize,xsize,False)[1][1]
            slices = imageRange_M(cen,ysize,xsize,False)
            CXMin = slices[1][0]
            slices[1][0] = CXMin + math.floor((LXMax - CXMin)/2)
            limits2[c][1][0] = math.floor((LXMax - CXMin)/2)
            limits.append((slice(*slices[0]),slice(*slices[1])))
        # non-top row, left-most tile just has top cropped
        elif c % xMon == 0:
            TYMax = imageRange_M(centres[c-xMon],ysize,xsize,False)[0][1]
            slices = imageRange_M(cen,ysize,xsize,False)
            CYMin = slices[0][0]
            slices[0][0] = CYMin + math.floor((TYMax - CYMin)/2)
            limits2[c][0][0] = math.floor((TYMax - CYMin)/2)
            limits.append((slice(*slices[0]),slice(*slices[1])))
        # the rest have top and right clipped
        else:
            LXMax = imageRange_M(centres[c-1],ysize,xsize,False)[1][1]
            slices = imageRange_M(cen,ysize,xsize,False)
            CXMin = slices[1][0]
            slices[1][0] = CXMin + math.floor((LXMax - CXMin)/2)
            limits2[c][1][0] = math.floor((LXMax - CXMin)/2)
            TYMax = imageRange_M(centres[c-xMon],ysize,xsize,False)[0][1]
            CYMin = slices[0][0]
            slices[0][0] = CYMin + math.floor((TYMax - CYMin)/2)
            limits2[c][0][0] = math.floor((TYMax - CYMin)/2)
            limits.append((slice(*slices[0]),slice(*slices[1])))
            
    # first put images in up to limits:
    for i,l in enumerate(zip(limits,limits2)):
        mergedIm[l[0]] = images[i][slice(*l[1][0]),slice(*l[1][1])].copy()
    
    # now crop or pad to output size
    if ysize_MFinal > ysize_m:
        mergedIm = np.pad(mergedIm,((ysize_MFinal - ysize_m,0),(0,0)))
    elif ysize_MFinal < ysize_m:
        mergedIm = mergedIm[ysize_m-ysize_MFinal:].copy()
    if xsize_MFinal > xsize_m:
        mergedIm = np.pad(mergedIm,((0,0),(xsize_MFinal - xsize_m,0)))
    elif xsize_MFinal < xsize_m:
        mergedIm = mergedIm[:,xsize_m-xsize_MFinal:].copy()
    
    return mergedIm




def noMergeStitch_7D(imData,centreList):
    
    """This is the main function that does everything
         imData has dimensions: 
         (times,regions,montages,zslices,channels,ysize,xsize)
         centreList is list of centres given by FindCentres.findCentres()
    """
    
    # extract imData dimensions:
    dims = imData.shape
    ysize,xsize = [dims[5],dims[6]]
    
    # make numpy array from centreList b/c lists are annoying to manipulate
    centres = np.array(centreList)
    
    # finding span of image centres:
    minCeny, minCenx = np.amin(centres,axis=0)
    maxCeny, maxCenx = np.amax(centres,axis=0)
    
    # initiate np.array for final image 
    # correct size calculated from dimsa and span of centres
    dims2 = list(dims)
    dims2[2] = 1
    dims2[5] = maxCeny - minCeny + ysize
    dims2[6] = maxCenx - minCenx + xsize
    mergedIm = np.zeros(dims2)
    
    # realign centres so they are indices of mergedIm
    minIndexY = minIndexFromCentre(minCeny,ysize)
    minIndexX = minIndexFromCentre(minCenx,xsize)
    centres = centres - np.array([minIndexY,minIndexX])
    
    # first make limits of the images, i.e. where the image ends 
    # and the merging begins
    #!!! for now we just use whole image slices
    limits = []
    for cen in centres:
        limits.append(imageRange_M(cen,ysize,xsize))
    #!!! TODO: check corners of limits of diagonal neighbours don't overlap
    # don't have to check horizontal or vertical neighbours, 
    # by definition their limits don't overlap
    
    # first put images in up to limits:
    for i,lim in enumerate(limits):
        mergedIm[:,:,0,:,:,lim[0],lim[1]] = imData[:,:,i].copy()
    
    #ToDo: do image placemnt properly with images stopping 
    # half way through the overlap
    
    return mergedIm



def imageRange_M(centre_M,ysize,xsize,returnSlice=True):
    """This returns a tuple containing 2 slices, 
        this tuple defines the indices covering an entire image
        i.e. it is the indices of the image with centre centre_M 
        In the coordinate system of M... given the tile sizes ysize and xsize
        It assumes convention that centre is the left pixel if size is even
    """
    
    # find y left limit
    y1 = int(centre_M[0] - np.floor((ysize-1)/2))
    # find y right limit
    y2 = int(centre_M[0] + np.ceil((ysize-1)/2)) + 1
    # make slice
    ySlice = [y1,y2,1]
    
    # do same for x
    x1 = int(centre_M[1] - np.floor((xsize-1)/2))
    x2 = int(centre_M[1] + np.ceil((xsize-1)/2)) + 1
    xSlice = [x1,x2,1]
    
    if returnSlice:
        return (slice(*ySlice),slice(*xSlice))
    else:
        return [ySlice,xSlice]



def minIndexFromCentre(centre,size):
    """This returns the minimum position of an image given its centre and size
        it is an int that can be used as an index
        i.e. it works whether the size is odd or even!
    """
    return int(centre - np.floor((size-1)/2))



def cenList2Size(centreList,ysize,xsize):
    """This finds the size in pixels of a montage which is made by placing 
        images of size (ysize,xsize) in the centre locations given by 
        centreList.
        I.e. it returns image dimensions (ysize_M,xsize_M)
    """
    minCeny, minCenx = np.amin(centreList,axis=0)
    maxCeny, maxCenx = np.amax(centreList,axis=0)    
    ysize_M = maxCeny - minCeny + ysize
    xsize_M = maxCenx - minCenx + xsize    
    
    return [ysize_M,xsize_M]
    