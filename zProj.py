import numpy as np
import math
from skimage.morphology import disk
from scipy.ndimage.filters import generic_filter
from skimage.transform import downscale_local_mean
from skimage.transform import resize



# normal maximum projection:
def maxProj(stack):
    return np.max(stack, axis=0)



def signalF(vals):
    return vals.mean()*vals.std()



# this is a function which finds where the signal is based on a local region (structuing element) within a slice
# the measure of signal is mean * std
# for each projection pixel it takes the pixel from the slice in the orginal image which had the most local signal
def fancyProj(stack,pixelSize,downscale):
    NZ,YSIZE,XSIZE = np.shape(stack)
    # radius wants to be about 30um so it is definitely bigger than nuclei
    radius = np.ceil(20/(pixelSize*downscale))
    selem = disk(radius)
    
    #initiate mask:
    stack2 = np.zeros((NZ,math.ceil(YSIZE/downscale),math.ceil(XSIZE/downscale)))
    stack3 = np.zeros((NZ,YSIZE,XSIZE))
    for i,im in enumerate(stack):
        stack2[i] = generic_filter(downscale_local_mean(im,(downscale,downscale)),signalF,footprint=selem,mode='reflect')
        stack3[i] = resize(stack2[i],(YSIZE,XSIZE))
    projection = np.zeros((YSIZE,XSIZE))
    for y in range(YSIZE):
        for x in range(XSIZE):
            projection[y,x] = stack[np.argmax(stack3[:,y,x]),y,x]
    
    return projection



# this selects N adjacent slices to take out
# selection is based on mean*STD signal measure
# it does a rolling average and selects N slices with minimum rolling average
def selectSlices_byMeanSTD_takeN(stack,N):
    #this is the signal measure, smaller for more signal:
    measures = [MAXSIGNALMEASURE/(im.mean()*im.std()) for im in stack]
    
    # rolling average is done by convolve:
    measures = np.convolve(measures, np.ones((N,))/N, mode='valid')
    
    # find min position and return the appropriate slice of stack
    minIndex = np.argmin(measures)  
    return stack[minIndex:minIndex + N]



# this one returns any slices where the measure is below a threshold
def selectSlices_byMeanSTD_thresh(stack,thresh):
    
    # just do list comprehension filtering:    
    return [im for im in stack if MAXSIGNALMEASURE/(im.mean()*im.std()) < thresh]



# this divides a stack into N*N smaller stacks
def sectioniseStack(stack,N,NZ):
    # first divide it up
    # look how np.array_split returns a list not a numpy array! (I guess because it can be jagged)
    sections = []
    for im in stack:
        sections.append(np.array_split(im,N))
        for n in range(N):
            sections[-1][n] = np.array_split(sections[-1][n],N,axis=1)
   
    # now need to reshape it so that each element of the list is a section with all z-slices
    # i.e. the thing returned is N*N long
    resections = [ [] for i in range(N*N) ]
    for n in range(N*N):
        for z in range(NZ):
            resections[n].append(sections[z][n%N][n//N])
    return resections



# takes a list of images which is N*N long and together form a N*N square image
# puts it back together
# watch out it doesn't work with a list of N*N image stacks! They have to be single images
def reassembleSections(imageList):
    N = int(math.sqrt(len(imageList)))
    columns = []
    for n in range(N):
        columns.append(np.concatenate(imageList[n*N:(n+1)*N]))
    image = np.concatenate(columns,axis=1)
    return image