import re
import os
import numpy as np
import tifffile
from datetime import datetime
from datetime import timedelta

from .exceptions import UnknownTifMeta

from skimage import measure
from scipy.ndimage import binary_fill_holes

# this takes a list of filenames and matches metadata to tif files
# i.e. makes a list [[metaname,[tifname1,tifname2,..]],...]
# it throws an exception if there's an andor tag at the end of your mfile
def groupSessionFiles(fnames):
    """ give this function a list of file names from saved data
    i.e. a list with mixed metadata files and tif files
    and it will return a list with files separated by session.
    Each element of list is in form: [metafile,[tiffile1,...tiffileN]]
    i.e. it finds matching tif files for each txt file in the list
    
    The sessions are sorted based on the start time of the metadata
    
    it raises an exception if the txt file has an Andor like tag
    i.e if it has something t0005 at the end
    because that would confuse it...
    because these tags are removed from the tif file in order to 
    do the matching.
    
    """
    
    # get all .txt files
    mFiles = [[file] for file in fnames if '.txt' in file]
    tFiles = [file for file in fnames if '.tif' in file]
    
    # now we set the order of the sessions in terms of the 
    # start time found in the metadata
    T0Reg = re.compile(r'Time=(\d\d:\d\d:\d\d)\n\[Created End\]')
    # start date regex:
    D0Reg = re.compile(r'\[Created\]\nDate=(\d\d/\d\d/\d\d\d\d)') 
    # format for datetime
    TX = '%d/%m/%Y %H:%M:%S'
    # loop over mFiles making a list with the associated start times
    mT0s = []
    for m in mFiles:
        # import metadata
        with open(m[0], 'rt') as theFile:  
            metadata = theFile.read()
        # extract start moment as datetime object
        startMom = meta2StartMom(metadata)
        mT0s.append(startMom)
    # use list of start times to sort mFiles:
    mFiles = [m for _,m in sorted(zip(mT0s,mFiles))]
    
    
    # this is the regex for detecting any tag at the end of filename: 
    tagRegex = r'_(f|m|t)\d{3,9}$'
    # error message (see assert below)
    eMess = 'metadata file {file} has a tag at the end which has the '\
            'same e.g. _t0005 format as Andor puts at the end of tif'\
            ' tif files. This makes it complex to automatically'\
            ' group metadata and tif files together and isn\'t'\
            ' handled yet. See generalFunctions.groupSessionFiles'\
            ' to make this work.'
    
    # find the tifnames for each metadata file
    for mf in mFiles:
        # metadata called e.g. data_t0002.txt aren't handled
        assert not re.search(tagRegex,mf[0][:-4]), eMess.format(file=mf)
        
        # strip tags off tiff files names and see if they match metafile name
        mf.append([tf for tf in tFiles if stripTags(tf)==mf[0][:-4]])
        
    return mFiles



def getSessionPaths(xPath,filters):
    """ This is similar to the method of XFold called getSessions()
    except it is a function by itself, not a method of a Class. 
    And it produces Session-like data, but not real Sessions.
    In fact it returns the [meta,[tif,tif...]] structure lists of file 
    paths that are produced by groupSessionFiles().
    This was motivated by wanting to get some session data during XFold 
    initialisation but not wanting to go round in cricles by using a method 
    of the class you were trying to set up an instance of. 
    It gets all the sessions found in the xPath
    It of course applies the filters, to filter out unwanted files
    """   
    # make a flat list of all the filepaths in XFold:
    walk = [x for x in os.walk(xPath)]
    fps = [os.path.join(x[0],fn) for x in walk for fn in x[2]]
    # filter out paths which have any terms included in filts
    fps = [fp for fp in fps if not any(fl in fp for fl in filters)]
    # pass fps->groupSessionFiles to organise into session structure:
    seshPaths = groupSessionFiles(fps)
    
    return seshPaths



def stripTags(filename):
    """ this removes the file extension and Andor tags off filenames.
    Andor tags are those things like _m0012.
    It takes any number of tags off the end.
    But doesn't delete any found in the middle.
    """ 
    # first get rid of extension
    filename = filename.split('.')[0]
    # this is the regex for detecting any tag, possibly repeated
    # ...which must appear at the end of filename:
    tagRegex = re.compile(r'(_(f|m|t|s)\d{3,9})+$')
    
    # return the filename rid of tags
    return tagRegex.sub('',filename)
                      
    
    
def tagIntFromTifName(name,findTag,stripTags=['t','f','m','s'],returnList=False):
    """This returns a list of ints taken from the tag that Andor puts on the 
        end of names of tif files when it splits data into different files. 
        The andor tag has a letter at the beginning, e.g. _t0005, here you 
        provide a list (stripTags) of which letters you want it to consider
        as counting as a tag. 
        It progressively strips these tags from the end of the filename 
        (after first having removed the .tif if it exists) and keeps each tag
        that has the letter that matches the letter provided in findTag.
        It returns the list of these tags converted to ints.
        I.e. name_m0004_name_m0001_t0005_m0109.tif will return
        [1,109] if you give it ['m'].
        """
    
    # take .tif off
    if name[-4:]=='.tif':
        fn = name[:-4]
    else:
        fn = name
    
    # make the regexes:
    findReg = r'_'+findTag+r'(\d{3,9})'
    stripEndReg = r'(_('+'|'.join(stripTags)+r')\d{3,9})$'
    
    # now keep stripping until non left but save findTags that crop up
    foundInts = []
    while re.search(stripEndReg,fn):
        foundTag = re.search(stripEndReg,fn).group(1)
        if re.search(findReg,foundTag):
            foundInts.append(int(re.search(findReg,foundTag).group(1)))  
        fn = re.sub(stripEndReg,'',fn)
    
    if returnList:
        return foundInts
    else:
        if len(foundInts)>0:
            return foundInts[0]
        else:
            return
    

def regexFromfilePath(reg,fPath,findAll=False,g=1,chars=10000,isFloat=False):
    """ this return regexes from fPath
        if the thing found is a digit char it always converts it to an int
        you can access other groups with g but default is g=1
        only loads no. 'chars' of characters from file in case too big
        for findAll = False you can use a compiled regex or a raw string
    """
    # loads data from fPath
    with open(fPath, 'rt') as theFile:  
        fileData = theFile.read(chars)            
    if findAll:
        N = re.findall(reg,fileData)
    else:
        if type(reg)==str:
            reg = re.compile(reg)
        N = reg.search(fileData)
        if N and type(N)!=list:
            N = N.group(g)
            if N.isdigit():
                N = int(N)
            if isFloat:
                N = float(N)
    return N

    

def buildXVectorList(xPath,xVectorList,filters):
    """ XVL is the user supplied XVectorList.
    We want it in the end to be a list of lists if experiment names
    A list of names for each session...
    ... with one name for each XY region in the session.
    But user might supply a path to txt file with XVL stored inside
    Or might have given nothing in which case we build our best guess
    """
    
    # if it's already a list there's nothing for us to do
    if type(xVectorList)==list:
        pass
    # if xVectorList is a string then it is... 
    # ...a path to a txt file storing the XVectorList
    elif type(xVectorList)==str and xVectorList != '':
        with open(xVectorList, 'rt') as theFile:  
            newXV = theFile.read(10000)
        xVectorList = [sesh.split(',') for sesh in newXV.split('\n')]
    # if xVectorList is empty then you put the default in
    elif xVectorList=='' or xVectorList==None:
        xVectorList = []
        sessionData = getSessionPaths(xPath,filters)
        reg = re.compile(r'XY : (\d+)\n')
        for sesh in sessionData:
            N = regexFromfilePath(reg,sesh[0])
            if N == None:
                xVectorList.append(['1'])
            else:
                xVectorList.append([str(x+1) for x in range(N)])
    return xVectorList
    
    

def chanDic(channel):
    """ This converts the channel's actual protocol 
    name to it's 'general name'.
    """
    channelDic = {'BF':'BF',
            'YFP':'YFP',
            'RFP':'RFP',
            'CFP':'CFP',
            'Tom_BF':'BF',
            'Tom_YFP':'YFP',
            'Tom_CFP':'CFP',
            'Tom_RFP':'RFP',
            'Label':'Label',
            None:None
                 }
    
    if channel not in channelDic.keys():
        errMess = 'WARNING: your channel name isn\'t in our channel '\
                    'dictionary. Please add it to the chanDic() function '\
                    'in XFH.py'
        print(errMess)
    
    return channelDic[channel]



def LUTDic(channel):
    """This takes a channel's 'general name' (see channelDic) and assigns 
    an LUT mix rule (see LUTMixer()).
    You might want to change this around or add new channels etc.
    """
    
    theLUTDic = {'BF':[True,True,True],
                 'YFP':[False,True,False],
                 'CFP':[False,False,True],
                 'RFP':[True,False,False],
                 'Label':[True,True,True]}
    
    if channel not in theLUTDic.keys():
        print('WARNING: your channel name isn\'t in our LUT dictionary. '\
              'Please add it to the LUTDic() function.')
    
    return theLUTDic[channel]



def LUTMixer(mixVector):
    """This makes an LUT in image j format from a boolean vector.
    I.e. you give it a vector like [True, Flase, True] to say which 
    RGB channels to put in the LUT.
    """
    val_range = np.arange(256, dtype=np.uint8)
    LUT = np.zeros((3, 256), dtype=np.uint8)
    LUT[mixVector] = val_range
    return LUT



def getProcessedDataDs(xPath,xSig):
    """This function looks in the parent directory of the path given for any
    directories containing the signature xSig in their name.
    It returns a set of paths to those directories.
    """
    
    # get parent directory path
    parPath = os.path.split(xPath)[0]
    # get the names of all objects in that directory
    listDir = os.listdir(parPath)
    # filter to get just the directories
    allDirs = [d for d in listDir if os.path.isdir(os.path.join(parPath,d))]
    # filter for directories must contain the signature xSig
    allDirs = [os.path.join(parPath,d) for d in allDirs if xSig in d]
    
    return set(allDirs)



def listStr2List(listString,convertNumeric=True):
    """This converts a string of a python list to a list.
    Only currently works for elements that are ints or 'None'
    """
    reg = r'\[(.*)\]'
    list1 = re.search(reg,listString).group(1)
    list1 = list1.split(',')
    
    list2 = []
    for l in list1:
        if l=='None':
            list2.append(None)
        elif l.replace(' ','').isdecimal() and convertNumeric:
            list2.append(int(l.replace(' ','')))
        elif l.replace('.','',1).isdecimal() and convertNumeric:
            list2.append(float(l))
        else:
            list2.append(l.replace('\'','').replace(' ',''))
    
    return list2



def maskFromOutlinePath(outLinePath):
    """ This takes the path of an image outline you have drawn and returns
        a binary mask with all values within the outline set to 
        1 and 0 elsewhere.
        The only requirement is that your outline has the pixel value that 
        is the highest in the image.
    """
    # import image
    with tifffile.TiffFile(outLinePath) as tif:
        outLine = tif.asarray()
    # normalise it
    outLine = outLine/np.max(outLine)
    # set non-maximum pixels to zero so we have a binary image
    outLine[outLine!=1.0] = 0
    # find connected components
    labels = measure.label(outLine)
    # find the connected component with the most pixels
    # we assume this is your outline
    biggestComponent = np.bincount(labels.flatten())[1:].argmax()+1
    # set everything to zero except your outline
    labels[labels != biggestComponent] = 0
    labels[labels == biggestComponent] = 1
    # fill in the outline
    mask = binary_fill_holes(labels)
    # return your mask
    return mask



def shapeFromFluoviewMeta(meta):
    """ This gets the 7D dimensions from fluoview metadata. 
        In some fluoview versions it doesn't include the dimension 
        name if the dimension size is 1 so we have to add it.
    """
    dims = meta['Dimensions']
    dimsDic = {l[0]:l[1] for l in dims}
    shapeKeys = ['Time','XY','Montage','Z','Wavelength','y','x']
    dims = []
    for k in shapeKeys:
        if k in dimsDic.keys():
            dims.append(dimsDic[k])
        else:
            dims.append(1)
    return dims



def tif2Dims(tif):
    """ This takes a tifffile.TiffFile object and returns dimensions 
        of the associated tifffile in the 7D format that we use. Currently
        works for fluoview and files we use in image j but we want to do 
        it for as many file types as possible.
    """
    fluo = 'fluoview_metadata'
    d = dir(tif)
    # for fluoview files:
    if fluo in d and tif.fluoview_metadata != None:
        meta = tif.fluoview_metadata
        dims = shapeFromFluoviewMeta(meta)                    
    # for image j ready files that we saved:
    elif ('imagej_metadata' in d and 
            tif.imagej_metadata != None and 
            'tw_nt' in tif.imagej_metadata.keys()):
        meta = tif.imagej_metadata
        baseString = 'tw_n'
        dimStrings = ['t','f','m','z','c','y','x']
        dims = [meta[baseString+L] for L in dimStrings]
    else:
        raise UnknownTifMeta()
        
    return dims



def meta2StartMom(meta):
    """ This takes a session's metadata file and returns a datetime object 
        of the moment when the file was started.
    """
    # the format of the datatime string we give it
    TX = '%d/%m/%Y %H:%M:%S'
    startTimeReg = re.compile(r'Time=(\d\d:\d\d:\d\d)\n\[Created End\]')
    # start date regex:
    startDateReg = re.compile(r'\[Created\]\nDate=(\d\d/\d\d/\d\d\d\d)')
    # delay reg, i.e. time between session starting and imaging starting
    delayReg = re.compile(r'Delay - (\d+) (\w+)')
    # take start moment from the vth metadata
    startT = re.search(startTimeReg,meta).group(1)
    startDate = re.search(startDateReg,meta).group(1)
    startMom = startDate + ' ' + startT
    startMom = datetime.strptime(startMom,TX)
    
    # add the delay time if necessary
    if re.search(startTimeReg,meta):
        delayT = int(re.search(delayReg,meta).group(1))
        if re.search(delayReg,meta).group(2)=='min':
            delayT = timedelta(minutes=delayT)
            startMom += delayT
        elif re.search(delayReg,meta).group(2)=='hr':
            delayT = timedelta(hours=delayT)
            startMom += delayT
        else:
            errMess = 'Your session had a delay before imaging but metadata'\
                    'is in an unknown format.'
             
    return startMom



def meta2TStep(meta):
    """ This takes a session's metadata file and returns a timedelta object
        of the time between time points.
    """
    # time interval group(1), units group(2)
    DTReg = re.compile(r'Repeat T - \d+ times? \((\d+) (\w+)\)')
    
    # find the time between time-points of this TData (from its 
    # parent session metadata)
    seshTStep = int(re.search(DTReg,meta).group(1))
    
    if re.search(DTReg,meta).group(2) == 'hr':
        seshTStep = timedelta(hours=seshTStep)
    elif re.search(DTReg,meta).group(2) == 'min':
        seshTStep = timedelta(minutes=seshTStep)
    if (re.search(DTReg,meta).group(2) != 'hr' and 
        re.search(DTReg,meta).group(2) != 'min'):
        errMess = 'the time step found in the metadata of your '\
        'data is not given in hrs or min. We don\'t handle '\
        'this yet so the labelling is going to go wrong! '
        raise Exception(errMess)

    return seshTStep



def onlyKeepChanges(theList):
    """ This makes a list from theList in which only elements which are 
    different from the previous are kept.
    """
    
    if len(theList)==0:
        return theList
    
    newList = []
    newList.append(theList[0])
    
    for l in theList[1:]:
        if l != newList[-1]:
            newList.append(l)
    
    return newList



def savedByXFoldQ(filepath):
    """ Returns True if the files was saved by this package, False otherwise
    Tests this by looking at metadata.
    """
    with tifffile.TiffFile(filepath) as tif:
        d = dir(tif)
        if ('imagej_metadata' in d and 
            tif.imagej_metadata != None and 
            'tw_nt' in tif.imagej_metadata.keys()):
            return True
        else:
            return False