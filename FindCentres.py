import numpy as np
from skimage.filters import sobel
from skimage.filters import gaussian
from skimage._shared.fft import fftmodule as fft



def findCentres(ims,NX,imageOverlap,method,parameters):
    """This function finds all the centres of a set of montage tiles that
    would make a well-aligned full image.
    It uses cross-correlation of just the overlapping regions to align things.
    ims is a list of all the images to be montaged, along with the channels... 
    so it has shape: (montages,channels,ysize,xsize).
    NX is the number of tiles in x dimension.
    imageOverlap is what the imaging overlap was (taken from metadata or 
    guess) (it will look at an image section a bit larger than this)
    It returns cenList_M which is a tuple of all the indices
    of the centre pixel of each image.
    The _M means the contents are in the final montage's coordinate system.
    It assumes all images are the same size.
    
    The parameters you give it are in the form: 
    [threshold,ampPower,maxShiftFraction,boxsize,sdt]
    
    threshold - threshold for 'enough signal', see lower functions
    ampPower - image is raised to this power at somepoint to amplify signal
            could increase for increase sensitivity?
    maxShiftFraction - maximum detected that is applied, as a fraction of 
                    image size b/c if a big shift is detected it is probably
                    wrong if detected shift is bigger it does 'auto' aligning
    boxsize - size of the box (in pixels) used in measuring signal with sobel
    sdt - standard dev of gaussian blur applied during signal measuring
    minSize - for images smaller than this it won't do cross-correlation
                alignment because the small size risks stupid results.
    """
    
    # unpack parameters
    # they get packaged again right away b/c they're only used in lower
    # functions but I unpackage them just so you can see them
    thresh = parameters[0]
    aPower = parameters[1]
    maxShift = parameters[2]
    bsize = parameters[3]
    sdt = parameters[4]
    minSize = parameters[5]
    cs = parameters[6]
    
    ysize,xsize = np.shape(ims[0,0])
    ycen = np.floor((ysize-1)/2).astype(int)
    xcen = np.floor((xsize-1)/2).astype(int)
    
    # package these parameters together so they don't bloat the script
    pars = (ysize,xsize,imageOverlap,method,thresh,
            aPower,maxShift,bsize,sdt,minSize)
    
    # initiate the main list we want to return
    cenList_M = []
    
    # main loop over all images:
    for j,im in enumerate(ims):
        
        # first image has centre local centre
        if j == 0:
            cenList_M.append(np.array([ycen,xcen]))
        # now do the top row where only the sides are compared:
        elif j < NX:
            cenList_M.append(findCen_S(ims[j-1],cenList_M[j-1],im,pars,cs))
        # then if it is the first colomn of a row you only compare the top:
        elif j % NX == 0:  
            cenList_M.append(findCen_T(ims[j-NX],cenList_M[j-NX],im,pars,cs))
        # or general case you compare the top and side:
        else:
            cenList_M.append(findCen_TS(ims[j-1],cenList_M[j-1],ims[j-NX],
                                        cenList_M[j-NX],im,pars,cs))
        
    return tuple(cenList_M)



def findCen_S(imageL,centreL_M,imageR,parameters,counts):
    """This function finds the centre of an imageR, knowing that it should sit 
    directly to the right of imageL and not aligning with any other tiles, 
    i.e. it is for the top row of montages, i.e. the _S stands for 'side only'
    imageL has centre pixel index given by centreL.
    Aligning is done by cross-correlation of just an overlap region.
    It returns the centre as a numpy array.
    
    parameters should be a tuple you pass in the form: 
    (ysize,xsize,imageOverlap,method,thresh,aPower,maxShift,bsize,sdt)
    
    ysize,xsize - the size in pixels of the image tiles being aligned.
    imageOverlap - your guess of what overlap the microscope sed (perhaps 
                taken from image metadata, as a fraction of image size.
    method - method could be lots of things but the only thing that matters
            in this part of the code is whether the string provided contains
            'noAlign' or not. If it contains no align then the 
            cross-correlation isn't done and the centres returned are just the
            'auto centres'. Auto aligning is just that calculated from 
            metadata overlap
    thresh - it searches for signal and does 'auto' aligning if not enough.
            this threshold defines 'enough signal'
    aPower - the image is raised to this power at somepoint to amplify signal
            could increase for increase sensitivity?
    maxShift - maximum detected that is applied, as a fraction of image size
            b/c if a big shift is detected it is probably wrong
            if detected shift is bigger it does 'auto' aligning
    bsize - size of the box (in pixels) used in measuring signal with sobel
    sdt - standard dev of gaussian blur applied during signal measuring
    """
    
    # unpack parameters from parameters:
    ysize,xsize = [parameters[0],parameters[1]]
    imageOverlap = parameters[2]
    method = parameters[3]
    thresh = parameters[4]
    aPow = parameters[5]
    maxShiftFrac = parameters[6]
    minSize = parameters[9]
    
    # unpack and re-pack the parameters for measureSig(image,pars)
    # unpack and repack just you you see them
    boxsize = parameters[7]
    sdt = parameters[8]
    pars = (boxsize,sdt)
    
    # initiate centreR so image R is directly against imageL (i.e. no overlap)
    # this way we will just add the shift to this centre later
    centreR_M = np.array([centreL_M[0], xsize + centreL_M[1]])    
    
    # the xsize of the section we will compare, 50% bigger thann the suggested
    # overlap 
    # this will be used as a number of pixels, i.e. a size not a index
    # remember we are using all of the y, and a section of the x
    testxSize = int(np.floor(1.2*imageOverlap*xsize))
    
    # these are the sections which will be compared
    # slicing like this gives images of size testxsize
    secL = imageL[:,:,xsize - testxSize:].copy()
    secR = imageR[:,:,:testxSize].copy()
    
    # calculate sobel signal in both sections, in all channels
    sigL = np.array([measureSig(secL[c],pars) for c in range(len(secL))])
    sigR = np.array([measureSig(secR[c],pars) for c in range(len(secR))])
    
    # test whether those signals are big enough and combine to make one 
    # channel selection: 
    chan = (sigL > thresh)*(sigR > thresh)
    
    # remove bad channels from signal b/c we use is as a multiplier next":
    sigL = sigL[chan]
    sigR = sigR[chan]
    
    # now delete channels which don't have enough signal in them  
    # amplify the channels by their signal measure so that the ones 
    # with signal are favoured during cross-correlation:
    secL = np.swapaxes(np.swapaxes(secL[chan],0,2)*sigL**aPow,0,2)
    secR = np.swapaxes(np.swapaxes(secR[chan],0,2)*sigR**aPow,0,2)
    
    # test if the sections contain signal, if not then just set the shift 
    # to what you'd expect from overlap:
    # also don't bother aligning if the images are small, you'll just 
    # get errors
    if (ysize < minSize or xsize < minSize or 'noAlign' in method):
        shift = np.array([0,int(xsize*imageOverlap)])
    elif any(chan)==False:
        shift = np.array([0,int(xsize*imageOverlap)])
        counts[0] += 1
    else:
        # calculate shift
        shift = register_translation(secL, secR)
        shift = shift.astype(int)
        
        #if shifts are too big then ignore them and set default:
        if shift[0] > maxShiftFrac*ysize:
            shift[0] = 0
            counts[1] += 1
        else:
            # the shift in y doesn't depend on the section you took:
            shift[0] = -shift[0]
        if shift[1] > maxShiftFrac*xsize:
            shift[1] = int(xsize*imageOverlap)
            counts[1] += 1
        else:        
            # the x shift of the actual image depends on the section you took!
            shift[1] = testxSize - shift[1] 

    return centreR_M - shift



def findCen_T(imageT,centreT_M,imageB,parameters,counts):
    """This function finds the centre of an imageB, knowing that it should sit 
    directly below of imageT and not aligning with any other tiles,
    i.e. it is for 2nd from top, far left tile only, _T stands for top.
    imageT has centre pixel index given by centreT.
    Aligning is done by cross-correlation of just an overlap region.
    It returns the centre as a numpy array.    
    
    parameters should be a tuple you pass in the form: 
    (ysize,xsize,imageOverlap,method,thresh,aPower,maxShift,bsize,sdt)
    
    ysize,xsize - the size in pixels of the image tiles being aligned.
    imageOverlap - your guess of what overlap the microscope sed (perhaps 
                taken from image metadata, as a fraction of image size.
    method - method could be lots of things but the only thing that matters
            in this part of the code is whether the string provided contains
            'noAlign' or not. If it contains no align then the 
            cross-correlation isn't done and the centres returned are just the
            'auto centres'. Auto aligning is just that calculated from 
            metadata overlap
    thresh - it searches for signal and does 'auto' aligning if not enough.
            this threshold defines 'enough signal'
    aPower - the image is raised to this power at somepoint to amplify signal
            could increase for increase sensitivity?
    maxShift - maximum detected that is applied, as a fraction of image size
            b/c if a big shift is detected it is probably wrong
            if detected shift is bigger it does 'auto' aligning
    bsize - size of the box (in pixels) used in measuring signal with sobel
    sdt - standard dev of gaussian blur applied during signal measuring    
    """    
    
    # unpack parameters from parameters:
    ysize,xsize = [parameters[0],parameters[1]]
    imageOverlap = parameters[2]
    method = parameters[3]
    thresh = parameters[4]
    aPow = parameters[5]
    maxShiftFrac = parameters[6]
    minSize = parameters[9]

    # unpack and re-pack the parameters for measureSig(image,pars)
    # unpack and repack just you you see them
    boxsize = parameters[7]
    sdt = parameters[8]
    pars = (boxsize,sdt)      
    
    # initiate centreB_M so imageB is directly against imageT (i.e. no overlap) 
    # this way we will just add the shift to this centre later
    centreB_M = np.array([centreT_M[0] + ysize, centreT_M[1]])  
    
    # the ysize of the section we will compare, 50% bigger than the 
    # suggested overlap
    # this will be used as a number of pixels, i.e. a size not a index
    # remember we are using all of the x, and a section of the y
    testySize = int(np.floor(1.2*imageOverlap*ysize))    

    # these are the sections which will be compared
    # slicing like this gives images of size testxsize
    secT = imageT[:,ysize - testySize:,:].copy()
    secB = imageB[:,:testySize,:].copy()
    
    # calculate sobel signal in both sections
    sigT = np.array([measureSig(secT[c],pars) for c in range(len(secT))])
    sigB = np.array([measureSig(secB[c],pars) for c in range(len(secB))])

    # test whether those signals are big enough and combine to make 
    # one channel selection: 
    chan = (sigT > thresh)*(sigB > thresh)    

    # remove bad channels from signal b/c we use is as a multiplier next":
    sigT = sigT[chan]
    sigB = sigB[chan]    
    
    # now amplify the channels by their signal measure so that the ones with 
    # signal are favoured during cross-correlation:
    secT = np.swapaxes(np.swapaxes(secT[chan],0,2)*sigT**aPow,0,2)
    secB = np.swapaxes(np.swapaxes(secB[chan],0,2)*sigB**aPow,0,2)
    
    # test if the sections contain signal, if not then just set the shift to 
    # what you'd expect from overlap
    if (ysize < minSize or xsize < minSize or 'noAlign' in method):
        shift = np.array([int(ysize*imageOverlap),0])
    elif any(chan)==False:
        shift = np.array([int(ysize*imageOverlap),0])
        counts[0] += 1
    else:
        # calculate shift
        shift = register_translation(secT, secB)
        shift = shift.astype(int)

        #if shifts are too big then ignore them and set default:
        if shift[0] > maxShiftFrac*ysize:
            shift[0] = int(ysize*imageOverlap)
            counts[1] += 1
        else:
            # the y shift of the actual image depends on the section you took!
            shift[0] = testySize - shift[0] 
        if shift[1] > maxShiftFrac*xsize:
            shift[1] = 0
            counts[1] += 1
        else:        
            # the shift in x doesn't depend on the section you took:
            shift[1] = -shift[1]         
    
    return centreB_M - shift



def findCen_TS(imageL,centreL_M,imageT,centreT_M,imageBR,pars,counts):
    """This function finds the centre of an imageBR, knowing it should sit
    directly below of imageT and to the right of imageL, 
    i.e. is for all tiles starting from 2nd tile of 2nd row. 
    _TS stands for top and side.
    imageL has centre pixel index given by centreL.
    imageT has centre pixel index given by centreT.
    Aligning is done by cross-correlation of just an overlap region.
    It returns the centre as a numpy array.    
    
    parameters should be a tuple you pass in the form: 
    (ysize,xsize,imageOverlap,method,thresh,aPower,maxShift,bsize,sdt)
    
    ysize,xsize - the size in pixels of the image tiles being aligned.
    imageOverlap - your guess of what overlap the microscope sed (perhaps 
                taken from image metadata, as a fraction of image size.
    method - method could be lots of things but the only thing that matters
            in this part of the code is whether the string provided contains
            'noAlign' or not. If it contains no align then the 
            cross-correlation isn't done and the centres returned are just the
            'auto centres'. Auto aligning is just that calculated from 
            metadata overlap
    thresh - it searches for signal and does 'auto' aligning if not enough.
            this threshold defines 'enough signal'
    aPower - the image is raised to this power at somepoint to amplify signal
            could increase for increase sensitivity?
    maxShift - maximum detected that is applied, as a fraction of image size
            b/c if a big shift is detected it is probably wrong
            if detected shift is bigger it does 'auto' aligning
    bsize - size of the box (in pixels) used in measuring signal with sobel
    sdt - standard dev of gaussian blur applied during signal measuring    
    """ 
    # unpack parameters from parameters:
    ysize,xsize = [pars[0],pars[1]]
    imageOverlap = pars[2]
    method = pars[3] 
    thresh = pars[4]
    aPow = pars[5]
    maxShiftFrac = pars[6]
    minSize = pars[9]

    # unpack and re-pack the parameters for measureSig(image,pars)
    # unpack and repack just you you see them
    boxsize = pars[7]
    sdt = pars[8]
    pars = (boxsize,sdt)    
    
    # start by aligning imageBR to imageL: 
    # (afterwards we will align to imageT and take an average)
    # initiate centreR so imageB is directly against imageL (i.e. no overlap)
    # this way we will just add the shift to this centre later
    centreBR_M = np.array([centreL_M[0], xsize + centreL_M[1]])  
    
    # the xsize of the section we will compare, 50% bigger than the 
    # suggested overlap
    # this will be used as a number of pixels, i.e. a size not a index
    # remember we are using all of the x, and a section of the y
    testxSize = int(np.floor(1.2*imageOverlap*xsize))    

    # these are the sections which will be compared
    # slicing like this gives images of size testxsize
    secL = imageL[:,:,xsize - testxSize:].copy()
    secBR = imageBR[:,:,:testxSize].copy()

    # calculate sobel signal in both sections
    sigL = np.array([measureSig(secL[c],pars) for c in range(len(secL))])
    sigBR = np.array([measureSig(secBR[c],pars) for c in range(len(secBR))])

    # test whether those signals are big enough and combine to 
    # make one channel selection: 
    chan1 = (sigL > thresh)*(sigBR > thresh)      

    # remove bad channels from signal b/c we use is as a multiplier next":
    sigL = sigL[chan1]
    sigBR = sigBR[chan1]    
    
    # now amplify the channels by their signal measure so that the ones 
    # with signal are favoured during cross-correlation:
    secL = np.swapaxes(np.swapaxes(secL[chan1],0,2)*sigL**aPow,0,2)
    secBR = np.swapaxes(np.swapaxes(secBR[chan1],0,2)*sigBR**aPow,0,2)    
    
    # test if the secs contain signal, if not then just set the shift 
    # to what you'd expect from overlap
    if (ysize < minSize or xsize < minSize or 'noAlign' in method):
        shift = np.array([0,int(xsize*imageOverlap)])
    elif any(chan1)==False:
        shift = np.array([0,int(xsize*imageOverlap)])
        counts[0] += 1
    else:
        # calculate shift
        shift = register_translation(secL, secBR)
        shift = shift.astype(int)
    
        # if shifts are too big then ignore them and set default:
        if shift[0] > maxShiftFrac*ysize:
            shift[0] = 0
            counts[1] += 1
        else:
            # the shift in y doesn't depend on the section you took:
            shift[0] = -shift[0]  
        if shift[1] > maxShiftFrac*xsize:
            shift[1] = int(xsize*imageOverlap)
            counts[1] += 1
        else:        
            # the x shift of the actual image depends on the section you took!
            shift[1] = testxSize - shift[1]       
    
    #shift the first estimation of centreBR_M:
    centreBR_M = centreBR_M - shift
    
    # now do everything again but aligning to imageT:
    # initiate centreR so imageB is directly against imageL (i.e. no overlap)
    # this way we will just add the shift to this centre later
    centreBR_M2 = np.array([ysize + centreT_M[0], centreT_M[1]])  
    
    # the xsize of the section we will compare, 50% bigger 
    # than the suggested overlap
    # this will be used as a number of pixels, i.e. a size not a index
    # remember we are using all of the x, and a section of the y
    testySize = int(np.floor(1.2*imageOverlap*ysize))    

    # these are the sections which will be compared
    # slicing like this gives images of size testxsize
    secT = imageT[:,ysize - testySize:,:].copy()
    secBR = imageBR[:,:testySize,:].copy()
    
    # calculate sobel signal in both sections
    sigT = np.array([measureSig(secT[c],pars) for c in range(len(secT))])
    sigBR = np.array([measureSig(secBR[c],pars) for c in range(len(secBR))])
    
    # test whether those signals are big enough and combine 
    # to make one channel selection: 
    chan2 = (sigT > thresh)*(sigBR > thresh)     

    # remove bad channels from signal b/c we use is as a multiplier next":
    sigT = sigT[chan2]
    sigBR = sigBR[chan2]        
    
    # now amplify the channels by their signal measure so that 
    # the ones with signal are favoured during cross-correlation:
    secT = np.swapaxes(np.swapaxes(secT[chan2],0,2)*sigT**aPow,0,2)
    secBR = np.swapaxes(np.swapaxes(secBR[chan2],0,2)*sigBR**aPow,0,2)    
    
    # test if the secs contain signal, if not then just set the 
    # shift to what you'd expect from overlap
    if (ysize < minSize or xsize < minSize or 'noAlign' in method):
        shift2 = np.array([int(ysize*imageOverlap),0])
    elif any(chan2)==False:
        shift2 = np.array([int(ysize*imageOverlap),0])
        counts[0] += 1
    else:    
        # calculate shift
        shift2 = register_translation(secT, secBR)
        shift2 = shift2.astype(int)

        #if shifts are too big then ignore them and set default:
        if shift2[0] > maxShiftFrac*ysize:
            shift2[0] = int(ysize*imageOverlap)
            counts[1] += 1
        else:
            # the y shift of the actual image depends on the section you took!
            shift2[0] = testySize - shift2[0] 
        if shift2[1] > maxShiftFrac*xsize:
            shift2[1] = 0
            counts[1] += 1
        else:        
            # the shift in x doesn't depend on the section you took:
            shift2[1] = -shift2[1]       
    
    #shift the second estimation of centreBR_M2:
    centreBR_M2 = centreBR_M2 - shift2
    
    #take average of the two centres:
    centreBR_M = (centreBR_M + centreBR_M2)/2
    
    return centreBR_M.astype(int)



def register_translation(src_image, target_im):
    """This is a version of the skimage register_translation
    like the skimage version, it does cross correlation of the image 
    fourier transforms ('fast fourier transform' algorithms)
    and returns the shifts calculated from the maxima of that.
    I've made it work for multi channel and removed the subpixel and error 
    stuff for simplicity.
    This takes images of dimensions: (channels,ysize,xsize)
    and sums the cross-correlations of channels before finding the max.
    """
    
    
    # images must be the same shape
    if src_image.shape != target_im.shape:
        raise ValueError("Error: images must be same size for "
                         "register_translation")
    
    src_freq = [fft.fftn(src_image[c]) for c in range(len(src_image))]
    target_freq = [fft.fftn(target_im[c]) for c in range(len(target_im))]

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq[0].shape
    im_product = [im1 * im2.conj() for im1, im2 in zip(src_freq,target_freq)]
    cross_correlation = [fft.ifftn(im) for im in im_product]
    
    # now sum the list of cross_correlations to find one max:
    cross_correlation = sum(cross_correlation)

    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq[0].ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts



def measureSig(image,pars):
    """A simple way to measure the amount of signal is to normalise by mean, 
    gaussian smooth, sobel and find the average. 
    This has a slight dependence on image size so we break into boxes of more
    constant size boxsize.
    image must be a 2D numpy array dimensions (ysize,xsize).
    """
    
    # unpack tha parameters:
    boxsize = pars[0]
    sdt = pars[1]
    
    ysize,xsize = image.shape
    # find the number of divisions that will give sizes closest to BOXSIZE:
    NY = round(ysize/boxsize)
    if NY == 0:
        NY = 1
    NX = round(xsize/boxsize)
    if NX == 0:
        NX = 1
    
    # do it all in one go, not very readable but this: 
    # splits in Y then in X to make a flat list where each element has
    # the sobel stuff applied then the mean of all is taken
    vals = []
    for ySplit in np.array_split(image,NY):
        for xySplit in np.array_split(ySplit,NX,axis=1):
            val = np.mean(sobel(gaussian(xySplit/np.mean(xySplit),sdt)))
            vals.append(val)
    
    return np.mean(vals)



def splitSignalDetect(image,N):
    """This function isn't used in the main script it is just 
    used for testing out the sobel signal measure
    image has shape (1,ysize,xsize)... i.e. stupid 
    sectioniseStack needs a z-stack...
    ...but reassembleSections needs shape (N*N,ysize,xsize), so we need 
    to give as zstack size 1
    this splits it into N*N squares
    sdt is the size of the blur we apply, it shouldn't be changed 
    without changing thresh
    thresh is the value of the signal measure that we count as a 
    positive result...
    ...hoping it's a constant that only depends on sdt, since we normalise 
    everything else by means
    """
    sdt = 4
    thresh = 0.035
    image = zProj.sectioniseStack(image,N,1)
    sigList = []
    #ave  = np.mean(image[0])
    print('no. of sections: ',len(image),'  no. of z: ',len(image[0]),
          '  dimensions of sections: ',image[0][0].shape)
    for i,im in enumerate(image):
        sig = np.mean(sobel(gaussian(im[0]/np.mean(im[0]),sdt)))
        #sig = np.mean(sobel(gaussian(im[0]/ave,sdt)))
        if sig > thresh:
            image[i][0][:,:] = np.ones(im[0].shape)
        else:
            image[i][0][:,:] = np.zeros(im[0].shape)
        if i%10==0:
            sigList.append(sig)
    image = [im[0] for im in image]
    print(np.mean(sigList))
    return zProj.reassembleSections(image)