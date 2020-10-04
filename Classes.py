"""
Here we define 4 classes which help globally organise and process the imaging
data obtained from a complex imaging experiment involving data with up to 
7 dimensions obtained in multiple imaging sessions. 

The 4 classes are: XFold, Session, TFile and TData. 
They are kind of heirarchical... 

XFold represents the folder where you put all the data. 
It contains the most global information. 

Session represents imaging sessions. I.e. one for every time you pressed 'go'
on the microscope resulting in a group of saved tif files with effectively 
one metadata.

TFile represents one tiff file inside your experiment folder. Importantly, 
it still doesn't contain the actual image data so a TFile object does not 
take a lot of memory. But it does store information about the contents of 
the tif file. It can be particularly useful for storing stuff that requires 
importing all the data, so is slow to get, but that you want to keep after 
you have thrown away the data (i.e. after you've freed space for loading 
the next data). 

TData represents the actual data inside a TFile. It is therefore the only 
one of these 4 classes that will take a lot of memory and you therefore 
should have a lot existing at once. It doesn't have to be all the data 
from the parent TFile but it will be recorded where within the TFile the 
data is taken from. All the data processing algorithms act upon these TData 
objects, i.e. are methods in this class.

You create an xfold with not much more than the XPath (= the path to your 
directory containing all the data). Then a method in the XFold class creates 
all the Sessions of that XFold. And a method in Session can create all the 
TFiles in a Session. And a method in TFile can create any desired TData from 
a TFile.

In the end, all the classes become connected. Each class stores the parent
class it was created from at the moment it was created 
(e.g. ParentSession.ParentXFold gives the XFold that the Session was 
created from). And the daughter objects are stored in the parent object with 
the exception of TDatas are not stored in TFiles because they would take too 
much data.

One feature of experiments that defines much of the structure of this 
code is that we have 7 image dimensions that behave differently w.r.t. the
global experiment structure:
Time, the 1st dimension, is always sequential along the Sessions. None of 
the other dimensions are.
The 2nd dimension, fields, can change between each session - only the user 
knows which field corresponds to which real experiment and should therefore
been grouped together.
Dimension 3,4,6 and 7 montages, z-slices, ysize and xsize, we assume for 
now that they are constant throughout your experiment.
Dimension 5 is channels and this can change too between sessions.

The different processing and analysis we want to do will change all these 
dimensions in different ways. 
Most severly, some processes change the data so that dimensions in the TData
no longer correspond to any that could be found in the TFile. 
E.g. z-projection makes a z-slice that doesn't exist in the TFile. 
Stitching does the same to montage tiles. 
Labelling does the same to channels and so can MatchChannels. 
But, crucially, a field in a TData will always correspond to a field in a 
TFile and the same for time (note, not all times and fields in allTFiles will
end up in TDatas, they can be deleted/ignored). This means we can have an 
object which contains where in a TFile you find a given time/field from a 
sTData and this object can always be used as an index.

Some vocabulary used:
The letter we use for a general unspecified axis is Q. 
The data a position q of this axis is a q-point.
I.e the q-point for:
time = time-point
field = field
montage = tile
Z = z-slice
C = channel

"""

toDo = 'Next: specify regions to not analyse.\n'
toDo += 'Next: Allow user specified ordeing of sessions,'\
      ' rather than time-based ordering.\n'
toDo += 'Next: equalise time frames method. (i.e. insert'\
      'repeats when the time difference between times is'\
      'double that of elsewhere.)\n'
toDo += 'Next: Do fancy stitching with merging overlaps. '
toDo += 'Message from LabelVideo: remember to check there '\
              'isn\'t a problem with the rounding.\n'

import os
import re
import numpy as np
from datetime import datetime
from datetime import timedelta
from itertools import product
from skimage.transform import downscale_local_mean
from skimage import io
from skimage.transform import resize

import tifffile

from . import generalFunctions as genF
from .zProj import maxProj,fancyProj
from .FindCentres import findCentres
from .noMergeStitch import noMergeStitch,noMergeStitch2,cenList2Size
from .LabelVideo import moment2Str,addTimeLabel
from .AlignExtract import findRegion,extractRegion
from .exceptions import UnknownTifMeta



class XFold:
    """This class represents an 'experiment folder'.
    That is a folder where you put all the data that you want the code to
    analyse and consider as one experiment.
    All time labelling, pixel normalisation cropping etc will be treating 
    all the images within as one set.
    So this object also contains all this global data.
    
    Attributes: 
    XPath - the path of the experiment folder
    XVectorList - each element is an ordered list of names of the fields
    Filters - files/directories including any of these terms in their name 
                are excluded from all analysis
    StartTimes - the path to a txt file containing the start data and time 
                that you want time labelling to start from for every fieldID
                in the XFold Sessions. The format should be:
                fieldID1: dd/mm/yyyy hh:mm:ss
                fieldID2: dd/mm/yyyy hh:mm:ss

    sizeDic - is a dictionary of {fieldID:[maxYSize,maxXSize]}
            The dimensions can change when you do the 
    
    xxxxxxx not sure we need this xxxxxxxjust use parameter in TFile?xxxxxxx
    fileTimes - fileTimes has an element for every Session
                that element is a list which has an element for every File
                that element is the no. of t-points found in the data of 
                that file when we run makeSessions() we put in the correct 
                no. of empty lists the each time we make run makeTFiles on
                a session the corresponding list gets updated.
    xxxxxxxxxxx maybe not this one either/ xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    maxDic - 
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    Methods:
    makeSessions(filts) - makes all sessions inside found XPath
                        except those with any element of filts in
                        their name (e.g. filter out montages). These sessions 
                        are stored inside the XFold but also the method 
                        returns a list containing those sessions so you can 
                        save that list separately.
    """
    
    
    # this needed for conversion of image types
    UINT16MAX = 65535
    
    # this is the string added to the start of any directories where 
    # the processed data is saved. It is also used to find any pre-existing 
    # processed data directories when you start building an XFold
    OutDirSig = 'XFoldAnalysis_'
    
    
    def __init__(self,XPath,XVectorList,Filters,StartTimes=None):
        self.XPath = XPath # the path to all the data and metadata
        self.Filters = Filters # files with these terms in name are ignored
        # we will build up a list of warnings during processing
        self.Warnings = [toDo]
        # see __doc__ for explanation of this one
        self.XVectorList = genF.buildXVectorList(XPath,XVectorList,Filters)
        # Label video just remembers the file path is there was one...
        self.StartTimes = StartTimes
        # ...the import contents of the file are in StartTimesDic:
        self.StartTimesDic = self.buildStartTimes()
        # contains all the sessions once they've been made
        self.SessionsList = []
        # have sessions been made already?
        self.sessionsMadeFlag = False
        # we make sessions here to fill the SessionsList but we also often 
        # run makeSessions in scripts to make our own list of sessions
        self.makeSessions()
        # a set of the known directories containing processed tiff files
        self.ProcessedDataDirs = genF.getProcessedDataDs(XPath,self.OutDirSig)
        self.ProcessedTFilesDic = {}
        self.makeProcessedTFiles(updateProcTFilesDic=True)
        
        
        # the followed are attributes that keep track of statistics during 
        # analysis, often from methods of TData
        self.SavedFilePaths = []
        self.BlankTimePointsCount = 0
        self.BlankTimePoints = []
        # the first one is for when there is not enough signal too align the 
        # tiles in a montage and the second is for when it's alignment gives 
        # a big shift (which we assume is erronous), in both cases we have 
        # done auto-alignment
        self.StitchCounts = [0,0]
        # counting how many images had uint16 overflows during homogenisation
        self.HomogOverflowCount = 0
        
        #xxxxxxxxxxxxxxxxxxxxx  maybe won't be making these after all  xxxxxxxxxxxxxxx
        #self.maxDic = {}
        #self.sizeDic = {}
        # fileTimes has an element for every Session
        # that element is a list which has an element for every File
        # that element is a list of indices which give the time position 
        # w.r.t the session
        # i.e. let element at position s of the list of sessions be a 
        # list called S[s]. 
        # The element at position f is a list I = S[s][f] element i is 
        # called I_i.
        # now we can say that the ith time point in fth file of the sth 
        # session of Xfold has contains the Ith timepoint of session s
        # index I at position i of the list for axis a here be 
        # called I_ia
        # then the images in position i of axis a in this data can be found 
        # in position I_ia along axis a of the parent Session
        # when we run makeSessions() we put in the correct no. of empty lists
        # the each time we make run makeTFiles on a session the corresponding
        # list gets updated
        #self.fileTimes = []
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  
    
    
    def buildStartTimes(self):
        """This loads the text file containing the start date and 
        time information and creates startTsDic.
        startTsDic is a dictionary containing all fieldIDs as keys and as 
        values the time-date (as datetime object) the experiment started 
        (from the user-provided file).
        
        StartTimes should be a string with the path to the text file 
        containing the data.
        The data can be in either of 2 formats. 
        Format 1:
        fieldID1: dd/mm/yyyy hh:mm:ss
        fieldID2: dd/mm/yyyy hh:mm:ss
        where fieldIDN are the fieldIDs given in the XVectorList.
        Every fieldID in XVectorList must have it's own start time 
        on a new line.
        Format 2:
        dd/mm/yyyy hh:mm:ss
        This one is interpreted as meaning every field started at the same
        given moment.
        
        It gives warnings if the StartTimes file start times aren't in 
        correct format or if there isn't one for every field.
        """
        
        if isinstance(self.StartTimes,str) and self.StartTimes != '':
            
            # raise exception if the path doesn't exist
            if not os.path.exists(self.StartTimes):
                errMess = 'You provided a string for the XFold\'s StartTimes'\
                        ' attribute. We iterpret that as being a path to a'\
                        ' .txt file containing start times for each field '\
                        'in your experiment. We could not find a file with'\
                        ' the path you provided so processing is stopping.'\
                        'You gave: {startTs}'
                raise Exception(errMess.format(startTs=self.StartTimes))
                
            # load data from file
            with open(self.StartTimes, 'rt') as theFile:
                labData = theFile.read()
            
            # this gets rid of any spurious whitespace at end of the .txt file
            endWSpaceReg = r'(\s*)$'
            labData = re.sub(endWSpaceReg,'',labData)
            
            # split the data by line:
            labData = labData.split('\n')
            # get rid of spurious white spaces at the end of lines:
            labData = [re.sub(endWSpaceReg,'',s) for s in labData]
            
            # start to build startTsDic
            startTsDic = [line.split(': ') for line in labData]
            # will need this: the set of all unique fieldIDs in XVectorList
            XVFields = set([y for x in self.XVectorList for y in x])
            
            # if just 1 line and there was no : in the line then we interpret 
            # as a single timedate for all fields, so we build that
            if len(startTsDic)==1 and len(startTsDic[0])==1:
                startTsDic = {k:labData[0] for k in XVFields}
            else:
                startTsDic = {k:v for k,v in startTsDic}
            
            # now just do checks that everything is good:
            # get the fieldIDs on the left of the colon:
            labRegs = set(list(startTsDic.keys()))
            
            
            # the regex for the start date and time format:
            regMom = r'\d{2}/\d{2}/\d{4} \d\d:\d\d:\d\d'
            # do all lines in the file match the required format?
            matchQ = all([re.match(regMom,x) for x in startTsDic.values()])
            # the format used in extracting datetime object from our string
            TX = '%d/%m/%Y %H:%M:%S'
            # if everything is ok then finalise startTsDic:
            if labRegs==XVFields and matchQ:
                startTsDic = {k:datetime.strptime(v,TX) 
                              for k,v in startTsDic.items()}
            # if there's a problem with the formats in the file then 
            # raise an exception
            elif not matchQ: 
                errMess = 'Warning: StartTimes file data '\
                            'isn\'t in correct format'
                raise Exception(errMess)
            # raise exception if the set of fieldIDs in file aren't exactly
            # the same as in the XVectorList
            elif labRegs!=XVFields:
                errMess = 'Warning: StartTimes fields don\'t '\
                            'match XVector fields'
                raise Exception(errMess)
        else:
            errMess = 'Warning: you didn\'t provide start times for the '\
            'fields in your experiments. We will assume the field started '\
            'at the beginning of the session that it first appeared.'
            self.Warnings.append(errMess)
            
            # the set of all unique fieldIDs found in XVectorList
            XVFields = set([y for x in self.XVectorList for y in x])
            # this is a dictionary of field name to index of session where it 
            # ...first appears
            field2firstSesh = {}
            for field in XVFields:
                for i,XV in enumerate(self.XVectorList):
                    if field in XV:
                        field2firstSesh.update({field:i})
                        break
            # for now we don't know the metadata of the sessions so we just 
            # leave it like this and update during makeSessions()           
            startTsDic = field2firstSesh
            
        return startTsDic    
        
        
        
    def makeSessions(self):
        """ This method makes all the sessions found in the XPath.
        It of course applies the Filters, to filter out unwanted files.
        All Sessions are saved within the XFold and also are returned 
        in a list so you could save them separately if you want.
        """
        
        # reset SessioList to blank list in case you have 
        # already made sessions
        self.SessionsList = []
        
        # make a flat list of all the filepaths in XFold:
        walk = [x for x in os.walk(self.XPath)]
        fps = [os.path.join(x[0],fn) for x in walk for fn in x[2]]
        # filter out paths which have any terms included in filts
        fps = [fp for fp in fps if not any(fl in fp for fl in self.Filters)]
        # pass fps->groupSessionFiles to organise into session structure:
        seshPaths = genF.groupSessionFiles(fps)
        
        # check XVector is correct length
        if len(seshPaths) != len(self.XVectorList):
            raise Exception("Your XVector wasn't the correct length!")

        # now make all the sessions
        allSesh = []
        allMeta = []
        for i,s in enumerate(seshPaths):
            #get the metadata of the session
            with open(s[0], 'rt') as theFile:  
                meta = theFile.read()
            # keep list of all metadatas b/c will need it to update startTsDic
            allMeta.append(meta)
            # check XVector for this session is correct length
            NFReg = Session.NFReg
            if re.search(NFReg,meta):
                NF = int(re.search(NFReg,meta).group(1))
                if NF != len(self.XVectorList[i]):
                    errMess = 'The XVector corresponding to session {sesh}.'\
                            ' didn\'t have the correct number of elements to'\
                            ' match the number of fields in the session.'
                    raise Exception(errMess.format(sesh=i))
            elif len(self.XVectorList[i])==1:
                # if it couldn't find no. of fields it's ok if there was only 
                #1 field because then meta won't have the bit that the regex 
                # searches for
                pass
            else:
                errMess = 'Couldn\'t find number of fields in metdata for '\
                            'session {sesh}. (Need new regex?)'
                raise Exception(errMess.format(sesh=i))
            
            # make the session and append to allSesh
            if isinstance(self.XVectorList,list):
                allSesh.append(Session(self,i,self.XVectorList[i],meta,s[1]))
            # maybe you haven't set up your XVectorList properly yet:
            # but let's try to never use this because it makes bad sessions
            # and I would definitely forget and cause problems later
            else:
                # let's print a warning as we know this is a bad thing to do
                print('Warning!: you\'re making sessions from an'\
                      ' XFold w/o good XVectors')
                allSesh.append(Session(self,i,None,meta,s[1]))                
        
        # update startTimeDic if we didn't complete it before 
        # (i.e. if there wasn't a txt file provided we are 
        # using metadata to guess)
        for k,v in self.StartTimesDic.items():
            if isinstance(v,int):
                startMom = genF.meta2StartMom(allMeta[v])
                self.StartTimesDic.update({k:startMom})
  
        # save the Sessions to the XFold's SessionsList
        self.SessionsList = allSesh
        # update sessionsMadeFlag
        self.sessionsMadeFlag = True
        
        return allSesh

    
    def makeProcessedTFiles(self,analDirPaths=None,updateProcTFilesDic=False):
        """This method creates all TFiles of the processed data found in 
        analDirPaths and returns them. Or if analDirPaths=None it looks 
        for directories with the xSig in the name and gets all those tif 
        files. 
        
        This will be a dictionary of analysisDirPath:[TFile1...TFileN].
        I.e. each analysisDirPath is a key.
        
        Normally we leave analDirPaths=None and use it to set the 
        Session.ProcessedTFilesDic.
        
        You can also give it the name of a directory and it makes TFiles of 
        all files in there. Can add this to Session.ProcessedTFilesDic if 
        you want too.
        """
        # will return this dictionary of analysisDirPath:[TFile1,TFileN... ]:
        allAnalDirs = {}
        
        # if user has provided a string we turn it to a list of length 1
        if isinstance(analDirPaths,str):
            analDirPaths = [analDirPaths]
        
        # if user didn't supply a list of analysis directories to use then
        # we look for any directories with the XFold analysis 'signature' 
        # in the parent folder of the xfold
        if not analDirPaths:
            xPath = self.XPath
            xSig = XFold.OutDirSig
            analDirPaths = list(genF.getProcessedDataDs(xPath,xSig))
        
        # loop over all analysis directories
        for adp in analDirPaths:
            # find all tif paths in that directory
            walk = [(dp,fn) for (dp,dn,fn) in os.walk(adp)]
            allTPaths = [os.path.join(dp,f) for (dp,fp) in walk for f in fp]
            allTPaths = [f for f in allTPaths if '.tif' in f]
            allTPaths = [f for f in allTPaths if genF.savedByXFoldQ(f)]
            
            # this sorts the files according to their s and t tags
            # otherwise it would be alphabetical which isn't reliable
            sTagsOfAll = [genF.tagIntFromTifName(tp,'s') for tp in allTPaths]
            tTagsOfAll = [genF.tagIntFromTifName(tp,'t') for tp in allTPaths]
            sortedPaths = sorted(zip(sTagsOfAll,tTagsOfAll,allTPaths))
            allTPaths = [tp for s,t,tp in sortedPaths]
            
            # these contain the lengths of the files for each tag
            # i.e. element with index i in this list = Q_i
            # then files with tag q_000i have Q_i Q-points 
            # (t-points/fields etc)
            Tag2FileLengthT = []
            Tag2FileLengthF = []
            Tag2FileLengthM = []
            
            allTFiles = []
            # cycle over all TFilesPaths in the session
            for tf in allTPaths:
                # get the dimensions of the TFile data from the metadata
                with tifffile.TiffFile(tf) as tif:
                    try:
                        _dims = genF.tif2Dims(tif)             
                        nt = _dims[0]
                        nf = _dims[1]
                        nm = _dims[2]
                        nz = _dims[3]
                        nc = _dims[4]
                    except UnknownTifMeta as error:
                        errMess = 'Error in makeProcessedTFiles() while '\
                                    'importing metadata from file {file}. '
                        print(errMess.format(file=tf),error)
                        raise
                
                # get the tag number (as an int) from the file name tags
                TTag = genF.tagIntFromTifName(tf,'t')
                FTag = genF.tagIntFromTifName(tf,'f')
                MTag = genF.tagIntFromTifName(tf,'m')
                
                # update the Tag2Length registers if it's a new tag
                if TTag!=None:
                    if TTag+1 > len(Tag2FileLengthT):
                        Tag2FileLengthT.append(nt)
                if FTag!=None:
                    if FTag+1 > len(Tag2FileLengthF):
                        Tag2FileLengthF.append(nf)
                if MTag!=None:
                    if MTag+1 > len(Tag2FileLengthM):
                        Tag2FileLengthM.append(nm)
                    
                # build the SeshQs of this TFile (see TFile for details)
                # they give the real Q-points of the TFile w.r.t to session
                T = [sum(Tag2FileLengthT[:TTag]) + i for i in range(nt)]
                F = [sum(Tag2FileLengthF[:FTag]) + i for i in range(nf)]
                M = [sum(Tag2FileLengthM[:MTag]) + i for i in range(nm)]
                Z = list(range(nz))
                C = list(range(nc))
                
                # this TFile isn't in the Session.TFileList:
                tFileN = None
                
                # find what session it came from
                seshTag = r'_s(\d+)'
                if re.search(seshTag,tf):
                    seshN = int(re.search(seshTag,tf).group(1))
                    seshTF = self.SessionsList[seshN]
                else:
                    errMess = 'Failed to make processed files dictionary for'\
                            ' xfold because there weren\'t session tags '\
                            '(e.g. _s0004) on the tiff files names in the '\
                            'processed data directory {dirc}.'
                    raise Exception(errMess.format(dirc=adp))
                
                # add the TFile to our list:
                allTFiles.append(TFile(seshTF,tFileN,tf,T,F,M,Z,C))
            
            # add the new entry of the dictionary
            allAnalDirs.update({adp:allTFiles})
            
        if updateProcTFilesDic:
            self.ProcessedTFilesDic.update(allAnalDirs)
        
        return allAnalDirs
    
    
    def BuildSummary(self):
        
        # start summary string that we will build up
        summary = ''
        
        # get all sessions and TFiles in the XFold
        allSesh = self.makeSessions()
        allTFiles = [TP.TPath for sesh in allSesh for TP in sesh.makeTFiles()]
        
        summary += 'Total no. of sessions: ' + str(len(allSesh)) + '\n'
        summary += 'Total no. of tiff files: ' + str(len(allTFiles)) + '\n'
        
        # size in memory
        totSize = sum([os.stat(tp).st_size for tp in allTFiles])/1000000
        summary += 'Total memory of tiff files: ' + str(totSize) + ' MB\n'
        
        totalNT = str(sum([s.SeshNT for s in allSesh]))
        summary += 'Total no. of time points (according to metadata): ' 
        summary += totalNT + '\n'
        uniqueF = str(len(set([y for x in self.XVectorList for y in x])))
        summary += 'Total no. of fields (no. of unique ID): ' + uniqueF + '\n'
        
        # total duration of experiment
        firstStart = genF.meta2StartMom(allSesh[0].Metadata)
        lastStart = genF.meta2StartMom(allSesh[-1].Metadata)
        timeDelta = genF.meta2TStep(allSesh[-1].Metadata)
        totT = lastStart - firstStart + timeDelta*allSesh[-1].SeshNT
        totD = str(totT.days)
        totH = str(totT.seconds//3600)
        totM = str(totT.seconds%3600//60)
        totT = totD + ' days, ' + totH + ' hours, ' + totM + ' minutes.'
        summary += 'Total time span: ' + totT + '\n'
        
        # NM,NZ and NC in 'set-like' form:
        summary += '\nThe following shows only the value of the given '\
                'attribute \nwhen it changes from one session to the next: \n'
        setM = str(genF.onlyKeepChanges([s.SeshNM for s in allSesh]))
        setZ = str(genF.onlyKeepChanges([s.SeshNZ for s in allSesh]))
        setC = str(genF.onlyKeepChanges([s.SeshNC for s in allSesh]))
        summary += 'Montage tiles: ' + setM + '\n'  
        summary += 'z-Slices: ' + setZ + '\n'  
        summary += 'number of channels: ' + setC + '\n'  
        
        # channel names
        setCNames = str(genF.onlyKeepChanges([s.SeshChan for s in allSesh]))
        summary += 'names of channels: ' + setCNames + '\n'  
        
        # session names:
        sNames = ''.join([s.Name+'\n' for s in allSesh])
        summary += '\nThe names of the sessions: \n' + str(sNames) + '\n'

        
        return summary
    
    
    def Summarise(self):
        summary = self.BuildSummary()
        print(summary)
    
    
    
    def printProcessingStats(self):
        """ Some of the methods of other classes (especially TData) will 
        report statistics on the things they have done which get saved 
        into the parent xfold's attributes. This method prints them, 
        typically to give a report at the end of a processing session.
        """
        # saved files
        print('Files saved from analysis of your xfold:')
        [print(e) for e in self.SavedFilePaths]
        # blank time points removed
        print('No of. blank time points removed: ',
              self.BlankTimePointsCount)
        print('Blank time point files: \n')
        [print(fp) for fp in self.BlankTimePoints]
        print('Number of auto-alinments during stitching '\
              'due to low signal: ',self.StitchCounts[0])
        print('Number of auto-alignments during stitching due '\
              'to large calculated shifts: ',self.StitchCounts[1])
        print('Number of images that had unit16 overflow during '\
              'division by filter during field of view homogensiation: ',
              self.HomogOverflowCount)
        warnings = set(self.Warnings)
        for w in warnings:
            print(w)
    
    
            
class Session:
    """ A Session object corresponds to one imaging session.
    An 'imaging session' is one run of an imaging protocol.
    I.e. it can have multiple tiff files but only one metadata txt file.
    Therefore this object stores the metadata (loaded as a string)
    and a list with all paths to the tiff files of the session.
    
    It deals with metadata, so we store all regexes here as class attributes.
    """
    
    
    def __init__(self,XFold,SessionN,XVector,Metadata,tFilePaths):
        self.ParentXFold = XFold # the parent XFold
        # the position you'll find this Session in parent XFold.SessionsList:
        self.SessionN = SessionN
        # The element of XVectorList that corresponds to this Session:
        self.XVector = XVector
        self.Metadata = Metadata # the imaging metadat
        self.tFilePaths = tFilePaths # all the file paths to associated tifs
        # give this session a name
        self.Name = genF.stripTags(os.path.split(self.tFilePaths[0])[1])
        self.TFilesList = [] # the list of all TFiles within this session
        self.TFilesMadeFlag = False # have the TFiles been made yet?
        # a dictionary of lists of TFiles which have already been processed
        # i.e. one item for each path in the set XFold.ProcessedDataDirs
        # only the ones where the data comes from this Session (even though 
        # the sessions have less meaning in the processed files because 
        # everything is put together)
        self.ProcessedTFilesDic = {}
        self.makeProcessedTFiles(updateProcTFilesDic=True)
        
        # the following session parameters tell us what is in the metadata...
        # i.e. they don't change as we process files, e.g. MatchChannels...
        # should be obvious b/c we don't change the files! We save new ones
        # the names of all the channels used here:
        self.SeshChan = re.findall(Session.chanReg,self.Metadata)
        
        # set the session imaging parameters and dimensions etc
        if re.search(Session.NTReg,self.Metadata):
            self.SeshNT = int(re.search(Session.NTReg,self.Metadata).group(1))
        else:
            self.SeshNT = 1
        if re.search(Session.NFReg,self.Metadata):
            self.SeshNF = int(re.search(Session.NFReg,self.Metadata).group(1))
        else:
            self.SeshNF = 1
        if re.search(Session.NMReg2,self.Metadata):
            self.SeshNM = int(re.search(Session.NMReg2,self.Metadata).group(1))
        else:
            self.SeshNM = 1
        if re.search(Session.NZReg,self.Metadata):
            self.SeshNZ = int(re.search(Session.NZReg,self.Metadata).group(1))
        else:
            self.SeshNZ = 1
        if re.search(Session.NCReg,self.Metadata):
            self.SeshNC = int(re.search(Session.NCReg,self.Metadata).group(1))
        else:
            self.SeshNC = 1
        if re.search(Session.NYReg,self.Metadata):
            self.SeshNY = int(re.search(Session.NYReg,self.Metadata).group(1))
        else:
            self.SeshNY = 1
        if re.search(Session.NXReg,self.Metadata):
            self.SeshNX = int(re.search(Session.NXReg,self.Metadata).group(1))
        else:
            self.SeshNX = 1

    
    # regexes for finding things from Andor metadata
    NTReg = re.compile(r'Time : (\d*)') # number of time points
    NFReg = re.compile(r'XY : (\d*)') # number of fields
    # number of montage tiles, can extract x (group(1)) and y (group(2))
    NMReg = re.compile(r'Montage Positions - \(\d* \( (\d*) by (\d*) \)\)')
    # just the total number:
    NMReg2 = re.compile(r'Montage : (\d*)')
    NZReg = re.compile(r'Z : (\d*)') # number of z-slices
    NCReg = re.compile(r'Wavelength : (\d*)') # number of channels
    NYReg = re.compile(r'y : (\d+) ') # y size
    NXReg = re.compile(r'x : (\d+) ') # x size
    # time interval group(1), units group(2)
    DTReg = re.compile(r'Repeat T - \d+ times? \((\d+) (\w+)\)')
    DZReg = re.compile(r'Repeat Z - (\d*) um in') # z-slice thickness
    chanReg = re.compile(r'\tChannel - (\w+)\n') # can get all names of chans
    # overlap of montage in %
    MOlapReg = re.compile(r'Montage=(Region|Edge)\tOverlap (\d*)') 
    # start time regex:
    startTimeReg = re.compile(r'Time=(\d\d:\d\d:\d\d)\n\[Created End\]')
    # start date regex:
    startDateReg = re.compile(r'\[Created\]\nDate=(\d\d/\d\d/\d\d\d\d)')
    # delay regex... i.e. time b/w start of protocol and start of imaging
    delayReg = re.compile(r'Delay - (\d+) (\w+)')
        
    
    def makeTFiles(self,setTFileList=True):
        """This method creates all TFiles in tFilePaths of the Session.
        It returns them in a list and saves them to the session 
        in self.TFilesList. Unless you set setTFileList=False in which 
        case it just returns them.
        
        Here is when we find the indices which give the positions of 
        the TFiles data w.r.t. the Session. This info is stored 
        in the TFiles. I.e. SeshT = [2,3] means this TFile contains the 3rd
        and 4th timepoint of the session.
        """ 
        
        # first set TFilesList to [] in case we have already made TFiles on 
        # this Session object
        if setTFileList:
            self.TFilesList = []
        
        # will return this list of TFiles:
        allTFiles = []
        # these contain the lengths of the files for each tag
        # i.e. element with index i in this list = Q_i
        # then files with tag q_000i have Q_i Q-points (t-points/fields etc)
        Tag2FileLengthT = []
        Tag2FileLengthF = []
        Tag2FileLengthM = []
        
        # cycle over all TFilesPaths in the session
        for i,tf in enumerate(self.tFilePaths):
            # get the dimensions of the TFile data from the metadata
            with tifffile.TiffFile(tf) as tif:
                try:
                    _dims = genF.tif2Dims(tif)             
                    nt = _dims[0]
                    nf = _dims[1]
                    nm = _dims[2]
                    nz = _dims[3]
                    nc = _dims[4]
                except UnknownTifMeta as error:
                    errMess = 'Error in makeTFiles() while importing '\
                                'metadata from file {file}. '
                    print(errMess.format(file=tf),error)
                    raise
                
            # get the tag number (as an int) from the file name tags
            TTag = genF.tagIntFromTifName(tf,'t')
            FTag = genF.tagIntFromTifName(tf,'f')
            MTag = genF.tagIntFromTifName(tf,'m')
            
            # unpdate the Tag2Length registers if it's a new tag
            if TTag!=None:
                if TTag+1 > len(Tag2FileLengthT):
                    Tag2FileLengthT.append(nt)
            if FTag!=None:
                if FTag+1 > len(Tag2FileLengthF):
                    Tag2FileLengthF.append(nf)
            if MTag!=None:
                if MTag+1 > len(Tag2FileLengthM):
                    Tag2FileLengthM.append(nm)
            
            # build the SeshQs of this TFile (see TFile for details)
            # they give the real Q-points of the TFile w.r.t to the session
            T = [sum(Tag2FileLengthT[:TTag]) + j for j in range(nt)]
            F = [sum(Tag2FileLengthF[:FTag]) + j for j in range(nf)]
            M = [sum(Tag2FileLengthM[:MTag]) + j for j in range(nm)]
            Z = list(range(nz))
            C = list(range(nc))
            
            allTFiles.append(TFile(self,i,tf,T,F,M,Z,C))
        
        # TFiles have been made!
        self.TFilesMadeFlag = True
        
        # put the TFiles inside this object
        if setTFileList:
            self.TFilesList = allTFiles
        
        return allTFiles
 

    def makeProcessedTFiles(self,analDirPaths=None,updateProcTFilesDic=False):
        """This method creates all TFiles of the processed data found in 
        analDirPaths and returns them. Or if analDirPaths=None it looks 
        for directories with the xSig in the name and gets all those tif 
        files. Only takes files correspondng to this session.
        
        This will be a dictionary of analysisDirPath:[TFile1...TFileN].
        I.e. each analysisDirPath is a key.
        
        Normally we leave analDirPaths=None and use it to set the 
        Session.ProcessedTFilesDic.
        
        You can also give it the name of a directory and it makes TFiles of 
        all files in there. Can add this to Session.ProcessedTFilesDic if 
        you want too.
        """ 
        # processed tifs belonging to this session will have this tag
        seshTag = '_s' + str(self.SessionN).zfill(4)
        
        # will return this dictionary of analysisDirPath:[TFile1,TFileN... ]:
        allAnalDirs = {}
        
        # if user has provided a string we turn it to a list of length 1
        if isinstance(analDirPaths,str):
            analDirPaths = [analDirPaths]
        
        # if user didn't supply a list of analysis directories to use then
        # we look for any directories with the XFold analysis 'signature' 
        # in the parent folder of the xfold
        if not analDirPaths:
            xPath = self.ParentXFold.XPath
            xSig = XFold.OutDirSig
            analDirPaths = list(genF.getProcessedDataDs(xPath,xSig))
        
        # loop over all analysis directories
        for adp in analDirPaths:
            # find all tif paths in that directory
            walk = [(dp,fn) for (dp,dn,fn) in os.walk(adp)]
            allTPaths = [os.path.join(dp,f) for (dp,fp) in walk for f in fp]
            allTPaths = [f for f in allTPaths if '.tif' in f]
            # filter for only sessions corresponding to this one
            allTPaths = [f for f in allTPaths if seshTag in f]
            
            # these contain the lengths of the files for each tag
            # i.e. element with index i in this list = Q_i
            # then files with tag q_000i have Q_i Q-points 
            # (t-points/fields etc)
            Tag2FileLengthT = []
            Tag2FileLengthF = []
            Tag2FileLengthM = []
            
            allTFiles = []
            # cycle over all TFilesPaths in the session
            for tf in allTPaths:
                # get the dimensions of the TFile data from the metadata
                with tifffile.TiffFile(tf) as tif:
                    try:
                        _dims = genF.tif2Dims(tif)             
                        nt = _dims[0]
                        nf = _dims[1]
                        nm = _dims[2]
                        nz = _dims[3]
                        nc = _dims[4]
                    except UnknownTifMeta as error:
                        errMess = 'Error in makeProcessedTFiles() while '\
                                    'importing metadata from file {file}. '
                        print(errMess.format(file=tf),error)
                        raise
                
                # get the tag number (as an int) from the file name tags
                TTag = genF.tagIntFromTifName(tf,'t')
                FTag = genF.tagIntFromTifName(tf,'f')
                MTag = genF.tagIntFromTifName(tf,'m')
                
                # update the Tag2Length registers if it's a new tag
                if TTag!=None:
                    if TTag+1 > len(Tag2FileLengthT):
                        Tag2FileLengthT.append(nt)
                if FTag!=None:
                    if FTag+1 > len(Tag2FileLengthF):
                        Tag2FileLengthF.append(nf)
                if MTag!=None:
                    if MTag+1 > len(Tag2FileLengthM):
                        Tag2FileLengthM.append(nm)
                    
                # build the SeshQs of this TFile (see TFile for details)
                # they give the real Q-points of the TFile w.r.t to session
                T = [sum(Tag2FileLengthT[:TTag]) + i for i in range(nt)]
                F = [sum(Tag2FileLengthF[:FTag]) + i for i in range(nf)]
                M = [sum(Tag2FileLengthM[:MTag]) + i for i in range(nm)]
                Z = list(range(nz))
                C = list(range(nc))
                
                # this TFile isn't in the Session.TFileList:
                tFileN = None
                
                # add the TFile to our list:
                allTFiles.append(TFile(self,tFileN,tf,T,F,M,Z,C))
            
            # add the new entry of the dictionary
            allAnalDirs.update({adp:allTFiles})
            
        if updateProcTFilesDic:
            self.ProcessedTFilesDic.update(allAnalDirs)
        
        return allAnalDirs

    
          
    def reDivideMFiles(self):
        """this method deals with the situation when andor separates tiffs 
        into _m000n files....our code doesn't want to bother with all that so
        just reDivides them into time and field files. 
        The important thing is to try not to end up with files that are too 
        big so it separates every time and field out of each m_file
        and puts together all the tiles from just one time and one field, 
        and saves that note this will still cause size problems if your zslice
        or channels or montages etc are too big.
        """
        
        # will need these regexes
        TReg = r'_t(\d{3,9})'
        FReg = r'_f(\d{3,9})'
        MReg = r'_m(\d{3,9})'
        TorFReg = r'(_f\d{3,9})|(_t\d{3,9})'
        
        # we will need the metadata and channel information too:
        meta = self.Metadata 
        chan = re.findall(Session.chanReg,meta)
        chan = [genF.chanDic(c) for c in chan]

        # get the total number of montage tiles from the filenames
        # ...assuming there is one called m0000 
        # ...and that they increase sequentially from there
        if all([re.search(MReg,p)==None for p in self.tFilePaths]):
            return
        elif any([re.search(MReg,p)==None for p in self.tFilePaths]):
            errMess = 'Some of your tFiles in session {session} had m-tags '\
                    'but others didn\'t. That\'s a bit strange you should '\
                    'have a look. The reDivideMFiles() method left it '\
                    'as it is.'
            print(errMess.format(session=self.SessionN))
            return        
        
        # make temporary directory for the fully separated files
        seshPath = os.path.split(self.tFilePaths[0])[0]
        tempPath1 = os.path.join(seshPath,'Mons1')
        
        # if this dir exists already there's a problem:
        errMess = 'directory used during reDivideMFiles (called {dir})'\
        'already exists, did you half stop half way through this analysis'\
        'before or something? All processing stopping, find and delete that'\
        'file and start again!'
        assert not os.path.exists(tempPath1), errMess.format(dir=tempPath1)
        
        # make the temp1 directory
        os.mkdir(tempPath1)
        
        monTags = [int(re.search(MReg,p).group(1)) for p in self.tFilePaths]
        nMon = max(monTags) + 1
        
        # loop through all tiff paths
        fileTsM = []      
        fileFsM = []
        for s,tp in enumerate(self.tFilePaths):
            with tifffile.TiffFile(tp) as tif:
                tifdata = tif.asarray()
                # we assume it is fluoview because we never save with m-tags
                metadata = tif.fluoview_metadata 
            
            # get dimensions from fluoview metadata
            _dims = genF.shapeFromFluoviewMeta(metadata)
            
            # put it in our 7D format
            for i,d in enumerate(_dims):
                if d == 1:
                    tifdata = np.expand_dims(tifdata,i)
            
            # take all the important numbers from the metadata
            dims = metadata['Dimensions'][0:7]
            NX,NY,NC,NZ,NM,NF,NT = [dim[1] for dim in dims]
            
            # update fileTsM 
            # we hope and assume that the t000N files are in good order
            if re.search(TReg, tp):
                fileT = int(re.search(TReg, tp).group(1))
                if len(fileTsM) < fileT+1:
                    fileTsM.append(NT)
                    
            # update fileFsM 
            # we hope and assume that the f000N files are in good order
            if re.search(FReg,tp):
                fileF = int(re.search(FReg,tp).group(1))
                if len(fileFsM) < fileF+1:
                    fileFsM.append(NF)
                    
            for t in range(NT):
                for f in range(NF):             
                    # save it:
                    # this makes a new 't000n' according to fileTsM 
                    # (i.e. accounting for many t-points in previous T files)
                    if re.search(TReg,tp):
                        T0 = sum(fileTsM[:int(re.search(TReg,tp).group(1))])
                        tTag = '_t' + str(t + T0).zfill(4)
                    else:
                        tTag = '_t' + str(t).zfill(4)
          
                    # this makes a new 'f000n' according to fileFsM 
                    # (i.e. accounting for many f-points in previous F files)
                    if re.search(FReg,tp):
                        F0 = sum(fileFsM[:int(re.search(FReg,tp).group(1))])
                        fTag = '_f' + str(f + F0).zfill(4)
                    else:
                        fTag = '_f' + str(f).zfill(4)        
                    
                    #remove any f|t tags from existing name and add new ones
                    tifNameM = re.sub(TorFReg, '',os.path.split(tp)[1][:-4])
                    tifNameM = tifNameM + tTag + fTag + '.tif'
                    
                    mon1OutPath = os.path.join(tempPath1,tifNameM)
                    tifffile.imsave(mon1OutPath,tifdata[t,f])
            
            # this takes a lot of memory so delete it
            del tifdata
        
        # now delete the original tif files (but not the metadata)
        for tf in self.tFilePaths:
            os.remove(tf) 
            
        # now we want to make a list of sublists where each sublist 
        # contains all m000n tifpaths for a given tTag_fTag:
        # all file names in the temp1 directory:        
        temp1Names = [f for f in os.listdir(tempPath1)]
        
        # gather filenames according to the tag we just added _t000n_f000n.tif
        tagsFilesDic = {}
        for fn in temp1Names:
            if fn[-16:] in tagsFilesDic.keys():
                tagsFilesDic[fn[-16:]].append(fn)
            else:
                tagsFilesDic.update({fn[-16:]:[fn]})
        mGatheredNames = list(tagsFilesDic.values())
                
            
        # now for each tTag_fTag, we load the full tiles set into one array and save it:
        outTPaths = []
        for mList in mGatheredNames:
            newtif = np.zeros((1,1,nMon,NZ,NC,NY,NX),dtype='uint16')
            for m,mName in enumerate(mList):
                with tifffile.TiffFile(os.path.join(tempPath1,mName)) as tif:
                    newtif[0,0,m] = tif.asarray()
            # delete the _m000n from the name of any mList name
            tifNameM = re.sub(MReg,'',mList[0])
            tifPathM = os.path.join(seshPath,tifNameM)
            # save the path so we can update self.tFilePaths later
            outTPaths.append(tifPathM)
            
            # this is the metadata for saving
            # see TData.SaveDataMakeTFile() for more info
            meta = {'hyperstack':'true','mode':'composite','unit':'um',
                    'spacing':'2.56','loop':'false','min':'0.0','max':'256',
                    'tw_NT':1,'tw_NF':1,'tw_NM':nMon,
                    'tw_NZ':NZ,'tw_NC':NC,'tw_NY':NY,
                    'tw_NX':NX,'tw_chan':str(chan)}
            # set ranges for each channel
            ranges = [x for i in range(4) for x in [0.0,65535.0]]
            ijmeta = {'Ranges':tuple(ranges)}
            # have to reshape into this shape for image j:
            dims = (1*1*nMon,NZ,NC,NY,NX)
            # do the save, reshaping the array for image j at the last moment
            tifffile.imsave(tifPathM,newtif.reshape(dims),imagej=True,
                            metadata=meta,ijmetadata=ijmeta)            
            
        # now delete the mon1 files and directory
        for fname in os.listdir(tempPath1):
            os.remove(os.path.join(tempPath1,fname))
        os.rmdir(tempPath1)
        
        # this changes session so that it is updated to newly saved files!
        self.tFilePaths = outTPaths
        
        return 
     
     
    def BuildSummary(self):
        
        summary = ''
        summary += 'Name of session: ' + self.Name + '\n'
        summary += 'No. of TFiles in session: '+str(len(self.TFilesList))+'\n'
        summary += 'TFiles in session: \n'
        [os.path.split(TF.TPath)[1]+'\n' for TF in self.TFilesList]
        
        summary += '\nSession channels: '+str(self.SeshChan)+'\n'
        summary += 'No. of time points: '+str(self.SeshNT)+'\n'
        summary += 'No. of fields: '+str(self.SeshNF)+'\n'
        summary += 'No. of montage tiles: '+str(self.SeshNM)+'\n'
        summary += 'No. of z-slices: '+str(self.SeshNZ)+'\n'
        summary += 'No. of channels: '+str(self.SeshNC)+'\n'
        summary += 'Size in Y: '+str(self.SeshNY)+'\n'
        summary += 'Size in X: '+str(self.SeshNX)+'\n'
        return summary
        
        
    def Summarise(self):
        summary = self.BuildSummary()
        print(summary)
    
    
        
class TFile:
    """ A TFile represents a tiff file but it still isn't the loaded data.
    There are methods to load data from this object though.
    It is not much more than the file path... except it does hold data about
    the contents of that file, data which might be time consuming to retrieve 
    each time if it requires opening all the data...
    But it's methods use the fastest ways to get that data.
    
    Session is the parent session
    
    NT,NF,NM,NZ,NC,NY,NX are the sizes of the dimensions of the data 
    inside the file.
    
    T,F,M,Z,C are each lists of indices which give the positions within 
    the parent Session where the data in the file comes from (w.r.t each axis)
    I.e. T = [2,3] mean the file contains the 3rd and 4th timepoint 
    of the Session.
    We decide to enforce this information when creating the TFile because 
    this is kind of the point of the whole package here: putting tiff 
    data in the global context of experiments.
    """
    
    
    def __init__(self,ParentSession,TFileN,TPath,SeshT,SeshF,SeshM,SeshZ,SeshC):
        self.ParentSession = ParentSession # the parent Session
        self.TFileN = TFileN # it's position in parent sessions's TFilesList
        self.TPath = TPath # the path to the TFile
        self.dimsFoundFlag = False
        # this functions fetches the shape (and all individual NQs) of the 
        # data within the TFile. As long as the file is a type we know we can 
        # get this info without loading all the data. Check it has worked 
        # with dimsFoundFlag()
        self.findFileDimensions()
        # this is an int taken from the number in the t-tag on the file name
        self.TTag = genF.tagIntFromTifName(self.TPath,'t')
        # ... same for f-tag
        self.FTag = genF.tagIntFromTifName(self.TPath,'f')
        # ...same for m-tag
        self.MTag = genF.tagIntFromTifName(self.TPath,'m')
        # these are each a list of indices giving the position of file images 
        # w.r.t the parent session.
        # i.e. let index I at position i of the list for axis a here be 
        # called I_ia
        # then the images in position i of axis a in this data can be found 
        # in position I_ia along axis a of the parent Session
        self.SeshT = SeshT
        self.SeshF = SeshF
        self.SeshM = SeshM
        self.SeshZ = SeshZ
        self.SeshC = SeshC
    
    
    def findFileDimensions(self):
        # load just metadata
        with tifffile.TiffFile(self.TPath) as tif:
            try:
                _dims = genF.tif2Dims(tif)             
                self.FileNT = _dims[0]
                self.FileNF = _dims[1]
                self.FileNM = _dims[2]
                self.FileNZ = _dims[3]
                self.FileNC = _dims[4]
                self.FileNY = _dims[5]
                self.FileNX = _dims[6]
                self.FileShape = _dims
                self.dimsFoundFlag = True 
            except UnknownTifMeta as error:
                errMess = 'Error in findFileDimensions() (perhaps during'\
                            'a TFile instance initiation) while importing '\
                            'metadata from file {file}. Your TFile\'s '\
                            'dimensions weren\'t set: things will probably '\
                            'go wrong!'
                print(errMess.format(file=self.TPath),error)  
                self.FileNT = None
                self.FileNF = None
                self.FileNM = None
                self.FileNZ = None
                self.FileNC = None
                self.FileNY = None
                self.FileNX = None
                self.FileShape = None                

    
    def makeTData(self,T='all',F='all',M='all',Z='all',C='all'):
        """ This method creates a specific TData from the TFile.
        You specify whichs frames (or range of frames) to make load
        load with T,F,M,Z,C etc. Either write 'all' or an int of the 
        one frame you want or provide a list of all frames you want.
        I.e. it doesn't accept slices yet.
        
        The dimension ordering of a TData is ALWAYS: 
        (times,fields,montages,zslices,channels,ypixels,xpixels)
        
        This so far works for images saved by Andor or by us using the 
        tifffile package.
        """
                  
        # first interpret the user input of index selections
        userSel = [T,F,M,Z,C]
        # turn into list length 1 if it's an int
        userSel = [[x] if isinstance(x,int) else x for x in userSel]
        # an f() to replace occurences of 'all' with a list of all indices
        all2Range = lambda p: list(range(p[1])) if p[0]=='all' else p[0]        
        # convert occurences of 'all' -> list of all indices
        userSel = list(map(all2Range, zip(userSel,self.FileShape[0:5])))
        # make an itertools product of the user selected indices so we
        # can cyle through every combination
        prod = product(*userSel)
        
        # this function converts 7D indices to their flat equivalent index
        # given the shapes of those domensions of course
        # actually it only does 5D of the 7D, XY ignored here
        def unravel(T,F,M,Z,C,NT,NF,NM,NZ,NC):
            return T*NF*NM*NZ*NC + F*NM*NZ*NC + M*NZ*NC + Z*NC + C
        
        # load the tiff pages that the user wants
        with tifffile.TiffFile(self.TPath) as tif:        
            # unravel our product with the FileShape to get index 
            # equivalents for the flat pages list
            pageIndices = [unravel(*X,*self.FileShape[0:5]) for X in prod]            
            # load the pages that the user asked for
            data = tif.asarray(key=pageIndices)
        
        # reshape to our standard 7D format shape
        dims = tuple([len(x) for x in userSel]+[self.FileNY]+[self.FileNX])
        data.shape = dims
        
        return TData(self,data,*userSel)
        
        
        
    def BuildSummary(self):
        
        summary = ''
        summary += 'TFile name: '
        summary += genF.stripTags(os.path.split(self.TPath)[1])+'\n'
        summary += 'From session: ' + self.ParentSession.Name + '\n'
        summary += 'TFile path: '+ self.TPath+'\n\n'
        summary += 'No. of time points: '+str(self.FileNT)+'\n'
        summary += 'No. of fields: '+str(self.FileNF)+'\n'
        summary += 'No. of montage tiles: '+str(self.FileNM)+'\n'
        summary += 'No. of z-slices: '+str(self.FileNZ)+'\n'
        summary += 'No. of channels: '+str(self.FileNC)+'\n'
        summary += 'Size in Y: '+str(self.FileNY)+'\n'
        summary += 'Size in X: '+str(self.FileNX)+'\n'
        return summary
        
        
    def Summarise(self):
        summary = self.BuildSummary()
        print(summary)
    
            
        
        
        
class TData:
    """ This class holds the actual image data 
    So these are the only objects in this package which take a lot of memory
    I.e. these are the only objects where you have to manage memory carefully
    The data is a numpy array. The dimension ordering is ALWAYS: 
    (times,fields,montages,zslices,channels,ypixels,xpixels)
    
    self.T,F,M,Z,C are the indices in the TFile that the data corresponds to.
    I.e. they are NOT the size of these dimensions.
    They are lists containing all the indices. Doesn't have to be consecutive 
    or ordering or non-repeating or anything... just a list of integers.
    They MUST be given when you create TData, otherwise you'd have to open 
    the file again to check so what would be the point in anything.
    I.e. there's no point in us allowing the creation of TDatas without this 
    info because lots of the methods wouldn't work. This whole package is 
    about putting tiffs together globally... if you want the methods 
    separately you can go to the generalFunctions.
    """
    
    def __init__(self,ParentTFile,data,FileT,FileF,FileM,FileZ,FileC):
        self.ParentTFile = ParentTFile
        self.data = data
        self.updateDimensions()
        
        # get the names of all the channels
        # we chose not to save meta to the object to remind ourselves that
        # the contents of meta apply to a whole session and are usually not
        # the same for this data object
        # if the TFile is processed then we get chan from the tw_chan 
        # inside the tiff file
        with tifffile.TiffFile(self.ParentTFile.TPath) as tif:
            if tif.imagej_metadata:
                _meta = tif.imagej_metadata
                self.chan = genF.listStr2List(_meta['tw_chan'])
            else:
                meta = self.ParentTFile.ParentSession.Metadata 
                self.chan = re.findall(Session.chanReg,meta)
        # convert them to the standard names
        self.chan = [genF.chanDic(c) for c in self.chan]
        # but you might not have take all channels from the TFile!:
        self.chan = [self.chan[fc] for fc in FileC]
        
        # chan might change but we should remember what they started as
        self.startChan = tuple(self.chan)
        
        # these Qlists have an element for each point on axis Q 
        # within this TData. The value of the element (it is an int) is
        # the position where you will find that Q-point along axis Q in 
        #the TFile. I.e. the first T-Point here might be the 4th in the
        # TFile if that is what we decided to load.
        # after manipulation you might be left with Q-points that don't 
        # correspond to any Q-point in the file so you set it to None
        self.FileT = FileT
        self.FileF = FileF
        self.FileM = FileM
        self.FileZ = FileZ
        self.FileC = FileC
    
    
    def updateDimensions(self):
        """This updates the record of the dimensions of the TData according
        to the shape that numpy finds. This shape is already the one we want
        because TFile.MakeTData does it for us.
        """
        dims = np.shape(self.data)
        self.NT = dims[0]
        self.NF = dims[1]
        self.NM = dims[2]
        self.NZ = dims[3]
        self.NC = dims[4]
        self.NY = dims[5]
        self.NX = dims[6]
        self.DataShape = dims
    
                
    def MatchChannels(self,endChans):
        """ this matches the channels of data to the user provided 
        tuple of channels called endChans.
        i.e. order channels as in user-supplied tuple endChans and add 
        blank channels if needed.
        """
        # if 1 of dim sizes is zero then there's no data, so return
        if np.prod(np.array(self.DataShape))==0:
            return
        
        # first add a blank channel at the end of the channels
        padTuple = ((0,0),(0,0),(0,0),(0,0),(0,1),(0,0),(0,0))
        self.data = np.pad(self.data,padTuple)
        # make foundChan list: position of endChan in self.chan
        foundChan = []
        for endChan in endChans:
            if endChan in self.chan:
                foundChan.append(self.chan.index(endChan))
            else: 
                # take from that blank channel we added on the end
                foundChan.append(-1)
        # reorder tifdata channels according to foundChannels
        self.data = self.data[:,:,:,:,foundChan,:,:].copy()
        # update self.chan 
        self.chan = list(endChans)
        # NC has probably changed so...
        self.updateDimensions()
        # FileC has also probably changed:
        self.FileC = [None if c==-1 else c for c in foundChan]
        
                
    def DownSize(self,downsize):
        """This method reduces downsizes the image data in x and y.
        Give an int for downsize to downsize x and y by this factor.
        Give a list [y,x] to downsize by different factors y and x.
        """
        
        # if 1 of dim sizes is zero then there's no data, so return
        if np.prod(np.array(self.DataShape))==0:
            return
        
        # make list if int downsize provided by user
        if isinstance(downsize,int):
            downsize = [downsize,downsize]
        # use skimage.transform: 
        self.data = downscale_local_mean(self.data,(1,1,1,1,1,*downsize))
        # NY and NX will have changed, so:
        self.updateDimensions()
        
                
    def DeleteEmptyT(self):
        """delete any time points that don't contain data
        """
        
        # if 1 of dim sizes is zero then there's no data, so return
        if np.prod(np.array(self.DataShape))==0:
            return        
        
        # find which time points have anything other than all zeros:
        nonEmptyTimes = [self.data[t].any() for t in range(self.NT)]
        self.data = self.data[nonEmptyTimes,:,:,:,:,:,:].copy()
        
        # update the parent xfold's records of this
        xfold = self.ParentTFile.ParentSession.ParentXFold
        xfold.BlankTimePointsCount += self.NT - sum(nonEmptyTimes)
        for i in range(self.NT - sum(nonEmptyTimes)):
            xfold.BlankTimePoints.append(self.ParentTFile.TPath)
        
        # NT may have changed, so:
        self.updateDimensions()
        self.FileT = [x for x,q in zip(self.FileT,nonEmptyTimes) if q]
        
        
    def Homogenise(self,HFileDic):
        """This helps correct non-uniform field of view problems.
        I.e. if there is a change of sensitivity across the field of view
        which is constant for all images then you can divide the image by an
        image of this non-uniformity. I.e take many images of a sample with 
        uniform fluorescence and make an average image.
        """  
        
        # if 1 of dim sizes is zero then there's no data, so return
        if np.prod(np.array(self.DataShape))==0:
            return        
        
        # need this to record overflow later
        parentXFold = self.ParentTFile.ParentSession.ParentXFold
        
        # import required filters (according to self.chan) HFilter images
        HFilts = [io.imread(HFileDic[c]) for c in self.chan]
        # make them the same size as self.data images
        HFilts = [resize(im,(self.NY,self.NX)) for im in HFilts]
        # make them float32s so you can divide by them
        HFilts = [im.astype('float32') for im in HFilts]
        # normalise HFilt so the max alue that you divide by is 1
        HFilts = [im/im.max() for im in HFilts]
        
        # loop over over image in self.data
        dims = [self.NT,self.NF,self.NM,self.NZ,self.NC]
        ranges = map(range,dims)
        for t,f,m,z,c in product(*ranges):
            # create a float32 of just that one image for division:
            _data = self.data[t,f,m,z,c].astype('float32')/HFilts[c]
            # add to count if pixels have become bigger than UINT16MAX
            if _data.max() > XFold.UINT16MAX:
                parentXFold.HomogOverflowCount += 1
            # convert back to uint16 to put in self.data
            self.data[t,f,m,z,c] = _data.astype('uint16')
            del _data
        
 
    def zProject(self,meth='maxProject'):
        """This does z-projection of the data.
        It can do normal maximum z-projection if you set meth='maxProject'
        Or it can do a fancy one that I created is you set: 
        meth=['signalDetect',downsizeFactor(?)]
        """
        
        # if 1 of dim sizes is zero then there's no data, so return
        if np.prod(np.array(self.DataShape))==0:
            return        
        
        # make temp np array to store data
        dims = (self.NT,self.NF,self.NM,1,self.NC,self.NY,self.NX)
        _data = np.zeros(dims,dtype='uint16')
        
        # maximum projection, just cycle through T,F,M and C...
        # this makes range(self.NT)... etc for itertools.product to work on
        ranges = map(range,[self.NT,self.NF,self.NM,self.NC])
        # method1: maximum projection
        if meth == 'maxProject':
            for t,f,m,c in product(*ranges):
                _data[t,f,m,0,c] = maxProj(self.data[t,f,m,:,c])
        # method2: the signal detection one I made:
        elif isinstance(zProject,list) and meth[0] == 'signalDetect':
            reg = r'x : (\d+) \* (\d*[.,]?\d*) : (\w\w)'
            p = genF.regexFromfilePath(reg,session[0],g=2,isFloat=True)
            for t,f,m,c in itertools.product(*ranges):
                _data[t,f,m,0,c] = fancyProj(self.data[t,0,m,:,c],p,meth[1])
        
        # we have to transfer _data contents to tifdataR without creating
        # a numpy 'view' (i.e. new array with 'base' of old array): use copy()
        self.data = _data.copy()
        del _data
        # dimensions have changed so:
        self.updateDimensions()
        # also the one remaining z-slice doesn't correspond to any 
        # in the parent TFile
        self.FileZ = [None]
      
        
    def StitchIt(self,method=''):
        """ This stitches montages together. There are several methods.
        
        The first thing is that because we are finding the best alignments 
        for each tile, we could end up with different sized images, both in
        the TData and across the whole XFold. To avoid this the stitch 
        functions do padding and/or cropping to the final image so that you 
        can define beforehand what size output you want. To make this work 
        across a whole XFold, this stiching method which gets applied to a 
        TData will look to the parent XFold to define the final size. That 
        way the size will be the same for all TData in the XFold.
        
        There are some paramters inside that you might want change one day
        Or make them changeable by the user.
        When aligning the montages we first search for signal, if there 
        isn't enough signal the aligning will do crazy things, so it does 
        'auto' aligning if not enough.
        Auto aligning is just that calculated from metadata overlap.
        
        The parameters you give it are in the form: 
        [threshold,ampPower,maxShiftFraction,boxsize,sdt]
        
        threshold - it searches for signal and does 'auto' aligning if 
                    not enough. This threshold defines 'enough signal'
        ampPower - the image is raised to this power at somepoint to 
                    amplify signal could increase for increase sensitivity?
        maxShiftFraction - maximum detected that is applied, as a fraction of
                            image size b/c if a big shift is detected it is
                            probably wrong if detected shift is bigger it does 
                            'auto' aligning
        boxsize - size of the box (in pixels) used in measuring 
                    signal with sobel
        sdt - standard dev of gaussian blur applied during 
                signal measuring
        minSize - for images smaller than this it won't do cross-correlation
                    alignment because the small size risks stupid results.
        """
        
        # nothing to do if there's not multiple tiles
        if self.NM ==1:
            return
        
        # will need these
        pSesh = self.ParentTFile.ParentSession
        pXFold = pSesh.ParentXFold
        
        # if 1 of dim sizes is zero then there's no data, so return
        if np.prod(np.array(self.DataShape))==0:
            return        
        
        # find whether this TData has been downsized
        # this will help choose a final montaged size that will match other
        # TData from the XFold
        seshNY = pSesh.SeshNY
        seshNX = pSesh.SeshNX
        downX = round(seshNX/self.NX)
        downY = round(seshNY/self.NY)
        
        # first decide what the output size will be
        # this is decided based on the whole XFold, so all montages 
        # of data from this XFold will have the same size
        # get all the values we need from all sessions in the XFold
        # but this gets a bit complicated because of downsize (not recorded)
        sessionsList = pXFold.SessionsList
        allMetas = [s.Metadata for s in sessionsList]
        OLReg = Session.MOlapReg
        allOverlaps = [int(re.search(OLReg,m).group(2))/100 for m in allMetas]
        NMReg = Session.NMReg
        allNY = [int(re.search(NMReg,m).group(2)) for m in allMetas]
        allNX = [int(re.search(NMReg,m).group(1)) for m in allMetas]
        NYReg = Session.NYReg
        NXReg = Session.NXReg
        allSizeY = [int(re.search(NYReg,m).group(1))/downY for m in allMetas]
        allSizeX = [int(re.search(NXReg,m).group(1))/downX for m in allMetas]
        # zip all these together
        allZipY = zip(allOverlaps,allNY,allSizeY)
        allZipX = zip(allOverlaps,allNX,allSizeX)
        
        # the size in x or y, call it q, would be given by:
        # (no. of tiles in Q)*(no. of pixels in Q) - overlapping part
        # with overlapping part given by: (N tiles - 1)*(N pixels)*overlap
        # we divide the overlap by 2 to give a number that is an overestimate
        # i.e. we hope we will always have some black padding at the edges:
        ySizeOut = [NMY*NY-(NMY-1)*NY*OL/2 for OL,NMY,NY in allZipY]
        xSizeOut = [NMX*NX-(NMX-1)*NX*OL/2 for OL,NMX,NX in allZipX]
        
        # we expect these to all be the same, otherwise you have changed 
        # montage or image size parameters half way through the experiment
        # which is going to cause problems
        if len(set(ySizeOut)) != 1 or len(set(xSizeOut)) != 1:
            errMess = 'Warning from StitchIt: The calculated output '\
                    'size of the montage is not the '\
                    'same for all session in you experiment folder. This '\
                    'means either the number of montage tiles along one '\
                    'axis, or the montage overlap, or the image size in '\
                    'pixels along one dimension has changed. This situation '\
                    'probably isn\'t handled properly by this code.'
            pXFold.Warnings.append(errMess)
        
        # now just take the maximum from each 
        # (which should be the one same value that they all have)
        ySizeOut = int(max(ySizeOut))
        xSizeOut = int(max(xSizeOut))
        # package them to stop it bloating code
        shapeOut = (ySizeOut,xSizeOut)
        
        # parameters to send to findCentres() (see __doc__)
        threshold = 0.035
        ampPower = 3
        maxShiftFraction = 0.1
        boxsize = 70
        sdt = 4
        minSize = 150
        cnts = pXFold.StitchCounts
        # package them so they don't bloat the code
        pars = [threshold,ampPower,maxShiftFraction,boxsize,sdt,minSize,cnts]
        
        # print a warning if we are skipping alignment due to small tiles
        if self.NY<minSize or self.NX<minSize:
            errMess = 'Warning: cross-correlation alignment was not used since '\
                        'the images were smaller than {minimumsize}. \'Auto\' '\
                        'align was used instead.'
            pXFold.Warnings.append(errMess)
        
        # get montage dimensions and image overlap from metadata:
        meta = self.ParentTFile.ParentSession.Metadata
        xMon = int(re.search(Session.NMReg,meta).group(1))
        olap = int(re.search(Session.MOlapReg,meta).group(2))/100
        
        # only want to align using these channels
        # i.e. no BF because cross-correlation doesn't work well
        fluoChans = ['YFP','CFP','RFP']
        # remove possible blank channels added by matchChannels
        fluoChans = [c for c in fluoChans if c in self.startChan]
        # now find indices of where fluoChans are in self.chans
        alignChans = []
        for c in fluoChans:
            if c in self.chan:
                alignChans.append(self.chan.index(c))
        # error message if no good channels for alignment:
        if alignChans==[]:
            errMess = 'No fluorescent channels found in data.'\
                      ' So we did noAlign'
            pXFold.Warnings.append(errMess)
            method += 'noAlign'
            alignChans.append(0)
        
        # initiate data to send to findCentres = sigIm 
        # sigIm is zproj of is 1 time, 1 field and only chans of alignChans
        # also centreLists storage array: one list (len=NM) for each t,f-point
        sigDims = (self.NM,len(alignChans),self.NY,self.NX)
        sigIms = np.zeros(sigDims,dtype='uint16')
        cenListsTF = np.zeros((self.NT,self.NF,self.NM,2),dtype='uint16')
        
        # build new sigIm for each t,f-point, 
        # we are careful with memory here, we build sigIms channel by channel
        # (trying to do it with numpy advanced indexing proved too complex)
        sizesTF = []
        for t in range(self.NT):
            for f in range(self.NF):
                for c in range(len(alignChans)):
                    sigIms[:,c] = self.data[t,f,:,:,c].copy().max(axis=1)
                cenListsTF[t,f] = findCentres(sigIms,xMon,olap,method,pars)
                
        del sigIms
        
        # initiate data for storage of final assemblies
        dims = (self.NT,self.NF,1,self.NZ,self.NC,ySizeOut,xSizeOut)
        _data = np.zeros(dims,dtype='uint16')
        
        # now assemble montages, again being careful with memory
        ranges = map(range,[self.NT,self.NF,self.NZ,self.NC])
        for t,f,z,c in product(*ranges):
            _data2 = self.data[t,f,:,z,c].copy()
            _data[t,f,0,z,c] = noMergeStitch2(_data2,cenListsTF[t,f],shapeOut,xMon)
            del _data2
        
        # delete,transfer,delete:
        self.data = _data.copy()
        del _data

        # set new montages,ysize,xsize       
        self.updateDimensions()
        # the tiles don't exist anymore, only on tile, so:
        self.FileM = [None]
        
        
    def LabelVideo(self,roundM=30):
        """This function adds time labels to the data.
            
        roundM is the minute interval that it will round to.
        """
        
        # if 1 of dim sizes is zero then there's no data, so return
        if np.prod(np.array(self.DataShape))==0:
            return        
        
        # this XFold attribute will be needed a lot:
        parentSession = self.ParentTFile.ParentSession
        startTimesDic = parentSession.ParentXFold.StartTimesDic
        # if there isn't a startTimesDic you can't do this
        if startTimesDic==None:
            print('No valid startdatetimes file was provided! '\
                  'Can\'t do labelling.')
        else:
            # you need a new blank channel for the label to go in:
            padDims = ((0,0),(0,0),(0,0),(0,0),(0,1),(0,0),(0,0))
            self.data = np.pad(self.data,padDims)
            # update attributes
            self.chan.append('Label')
            self.NC += 1
            self.FileC.append(None)
                        
            # get this TDatas metadata
            meta = self.ParentTFile.ParentSession.Metadata
            
            # the session of this TData started at this moment:
            seshStartMom = genF.meta2StartMom(meta)
            # find the time between time-points of this TData
            seshTStep = genF.meta2TStep(meta)
            
            # the first time in this TData probably isn't the first 
            # in its parent session. This gives the timedelta between the 
            # start of the session and the first time-poj=int of this TData
            TDataShift = (self.FileT[0] + self.ParentTFile.SeshT[0])*seshTStep
            
            # find time since field's experiment started for all t and f of 
            # data. f nested within t to make a list of lists of timedeltas
            tStrings = []
            for i,t in enumerate(self.FileT):
                tStrings.append([])
                for f in self.FileF:
                    
                    # find the moment when frame was taken:
                    # this first no. is no. of time-points since session start
                    tsSinceSeshStart = self.ParentTFile.SeshT[t]
                    frameTakenMom = seshStartMom + seshTStep*tsSinceSeshStart
                    
                    # the index of this field w.r.t the fields of the session
                    seshF = self.ParentTFile.SeshF[f]
                    fieldID = self.ParentTFile.ParentSession.XVector[seshF]
                    fieldStartMom = startTimesDic[fieldID]
                    
                    # this is the time we want to print on the image:
                    # it is the time between when the frame was taken and 
                    # when the field's experiment started
                    tSinceFdStart = frameTakenMom - fieldStartMom
                    tSinceFdStart = moment2Str(tSinceFdStart,roundM)
                    # add it to the list of lists we're making:
                    tStrings[i].append(tSinceFdStart)
             
            # add the time string to the label channel:
            dims = [self.NT,self.NF,self.NM,self.NZ]
            ranges = map(range,dims)
            for t,f,m,z in product(*ranges):
                addTimeLabel(self.data[t,f,m,z,-1],tStrings[t][f],10)
    
    
    def AlignExtract(self,templateDic,deltaAng=0.25,maxAng=15,manualScale=False):
        """This extracts the regions of your data corresponding to the 
        templates you give it.
        The templates should be rectangular BF images of the region 
        you want to extract.
        For each field, the template in fact should be a maximum projection 
        (because this we do that to the data to make the search) and it
        should be from a time point near the middle of the experiment 
        (because that will minimise the maximum difference in image it 
        has to match.)
        Save your templates somewhere as tifs and pass them to this 
        function via templateDic.
        
        It searches for your template using cross-correlation with 
        translation and rotation.
        But since this takes time and we know there should be much shift 
        we set a maximum angle (in degrees), usually something less 
        than 10 is good. 
        We also pass it a deltaAng, i.e. the steps in angle 
        that will be tried.
        
        If the template size varies between fields, the output will be 
        padded because numpy arrays can't hold jagged arrays, i.e. we can't
        have fields of different size stored in the TData.
        
        This assumes you are extracting from a stitched data point! 
        i.e. montages = 1
        It also assumes the template you saved is a max-projection 
        of the BF channel.
        """
        
        # ToDo warning
        print('ToDo warning: you should downsize the templates '\
              'automatically by havin pixel size attributes, i.e. '\
              'pixel=xum, which can be compared between the template '\
              'and TData to find how much scaling needs to be done.')
        
        # if 1 of dim sizes is zero then there's no data, so return
        if np.prod(np.array(self.DataShape))==0:
            return        
        
        # package these to not bloat code
        ps = (deltaAng,maxAng)
        
        # first import all templates and find the max y and max x size
        maxYSize = []
        maxXSize = []
        for tem in templateDic.values():
            with tifffile.TiffFile(tem) as tif:
                template = tif.asarray()
            # downscale template if needed
            if manualScale:
                if isinstance(manualScale,int):
                    manualScale = tuple([manualScale,manualScale])
                if isinstance(manualScale,list):
                    manualScale = tuple(manualScale)
                template = downscale_local_mean(template,manualScale)
            ysizeT,xsizeT = template.shape
            maxYSize.append(ysizeT)
            maxXSize.append(xsizeT)
        maxYSize = max(maxYSize)
        maxXSize = max(maxXSize)
        shapeMx = (maxYSize,maxXSize)
        
        
        # make a temporary array to store all data in
        dims = (self.NT,self.NF,1,self.NZ,self.NC,maxYSize,maxXSize)
        _data = np.zeros(dims,dtype='uint16')
        
        # cycle over field and then times because each field has a 
        # different template to be loaded and each time needs aligning 
        # (we find the alignment only to z-projected BF channel so we 
        # don't need to loop over z-slices and channels here)
        for i,f in enumerate(self.FileF):
            # first get the fieldID of the current field:
            seshF = self.ParentTFile.SeshF[f]
            fieldID = self.ParentTFile.ParentSession.XVector[seshF]
            
            # import the template
            with tifffile.TiffFile(templateDic[fieldID]) as tif:
                template = tif.asarray()            
            # it will need to be a float
            template = template.astype('float32')
            # downscale template if needed
            if manualScale:
                if isinstance(manualScale,int):
                    manualScale = tuple([manualScale,manualScale])
                if isinstance(manualScale,list):
                    manualScale = tuple(manualScale)
                template = downscale_local_mean(template,manualScale)
            # the shape of the template
            shape = template.shape
            
            
            # where to find the bright field in the channels axis:
            if 'BF' not in self.chan:
                errMess = 'We couldn\'t find a channel called BF in your '\
                        'data. You need a bright field channel to extract'\
                        'from so this isn\'t going to work now!'
                print(errMess)
            else:
                BFIndex = self.chan.index('BF')
            
            for t in range(self.NT):
                # make a maximum projection of chan BF to do the search in
                BFZProj = self.data[t,i,0,:,BFIndex].copy().max(axis=0)
                BFZProj = BFZProj.astype('float32')
                ang,shift = findRegion(BFZProj,template,*ps)
                qs = (ang,shift)
                _data[t,i] = extractRegion(self.data[t,i],*qs,*shape,*shapeMx)
                
        self.data = _data.copy()
        del _data
        self.updateDimensions()
    
    
    def SaveDataMakeTFile(self,outDir=None,overWrite=False,
                          allowDefault=False,firstInBatch=None):
        """This takes the image data from your TData and saves it in a way 
        that image j will be able to open it and have all the channel and 
        hyperstack info it needs.
        
        It ALWAYS separates fields into different files AND folders.
        
        outDir: Where to put the data you save.
        This all uses the output data directory name signature 
        XFold.OutDirSig = xSig. 
        I.e. that is a prefix added to any directories containing data 
        output by this method.
        You can give it a single word and it will put data in a directory 
        called xSig+outDir which will be put in the parent directory of 
        the associated XFold.XPath.
        If you give it a path, they will all be put in there. 
        The xSig won't be added anywhere so this data might not be 
        automatically found if you try to do further analysis on the data.
        You give can give it a dictionary of outDirectories which gives 
        the path for each field. 
        If you don't provide anything then it saves to
        os.path.join(xPParentD,xSig+N,'Exp'+fieldID), where N is the 
        number of XFold.ProcessedDataDirs items.
        xPParentD = XPath's parent directory
        
        It gives images j the correct LUTs according to TData channels.
        
        It strips old tags from the file name and adds new tags to
        all names. We have our own tag system, _s000N_t000M for the session 
        the data comes from and the time point relative to that session.
        
        It returns a TFile related to this new TData in case you want to do 
        some more processing afterwards.
        
        """

        # if 1 of dim sizes is zero then there's no data, so return
        if np.prod(np.array(self.DataShape))==0:
            errMess = 'The processed data of file: {file} was not saved '\
                    'because there was no data left after processing! '\
                    '(Probably it was a blank data point removed by '\
                    'deleteEmptyTs())'
            # we decide not to print this, equivalent should be save in 
            # xfold stats, in self.BlankTimePoints
            #print(errMess.format(file=self.ParentTFile.TPath))
            return        
        
        noBatchErrMess = 'You have not provided a name for the directory to '\
                    'save the data in. You have allowed for default naming '\
                    'of the directory but you haven\'t told us if this is '\
                    'the first TData to be saved in the batch or not. We '\
                    'can\'t find this from the data itself because '\
                    'sometimes the first time or field from the XFold will '\
                    'be excluded from analysis (e.g. deleted time point '\
                    'because it\'s blank), so you have to tell us. All '\
                    'processing is stopping.'
        
        noDefaultErrMess = 'You have not given a name for the directory to '\
                    'save in AND you have not set allowDefault=True so '\
                    'we have no way of deciding where to save the data. '\
                    ' All processing is stopping.'

        # these are some class attributes we'll use a lot   
        pSesh = self.ParentTFile.ParentSession
        pXFold = pSesh.ParentXFold
        # the signature for the out directory name, 
        #i.e. the default prefix for the directory name
        xSig = XFold.OutDirSig 
        # the parent directory of the XFold:
        parD = os.path.split(pXFold.XPath)[0]
        
        # now we make the analysis directory path from user-supplied outDir
        # if the user didn't provide outDir...
        if not outDir:
            # if they've told us we can make up the default name...
            if allowDefault==True:
                # if 1st in batch then the name we are about to make isn't
                # in the list of directories yet so we add 1 to the ID 
                # number that forms the name
                if firstInBatch:
                    outDirI = len(pXFold.ProcessedDataDirs) + 1
                # if not first in batch then don't add one
                elif firstInBatch==False:
                    outDirI = len(pXFold.ProcessedDataDirs)
                # if they forgot to say if this is 1stInBatch then: prob
                elif firstInBatch==None:
                    raise Exception(noBatchErrMess)                        
                # now make the analPath:
                analDir = xSig+str(outDirI).zfill(3)
                analPath = os.path.join(parD,analDir)
            # if we're not allowed to make up default there's a problem!
            else:
                raise Exception(noDefaultErrMess)     
        # if the user provided a string then we use this for the analDir
        elif isinstance(outDir,str):
            # if they provide a path with 2 levels or more then we assume 
            # they want to give the parent directory too
            if os.path.split(outDir)[0]:
                parnt = os.path.split(out)[0]
                analPath = os.path.join(parnt,xSig+os.path.split(out)[1])
            # if they've provide just a single word, not a path with more 
            # than one level, we set the parent directory of the XFold as 
            # the parent directory of the output folder
            else:
                analPath = os.path.join(parD,xSig+outDir)
        # if not string then there's a problem    
        else:
            raise Exception('outDir is not is a form we can deal with!')
        
        # now analPath is definitely made we can add it to the set
        # of analsyis directories
        pXFold.ProcessedDataDirs.add(analPath)
        # and add item with blank list this to session's
        # ProcessedTFilesDic if not already there: 
        if analPath not in pSesh.ProcessedTFilesDic.keys():
            pSesh.ProcessedTFilesDic.update({analPath:[]})
        # make the directory if it doesn't exist yet
        if not os.path.exists(analPath):
            os.mkdir(analPath)        
        
        allTFiles = []
        # cycle through fields because each will be saved separately
        for i,f in enumerate(self.FileF):        
            
            # what is our fieldID...
            seshF = [self.ParentTFile.SeshF[f]]
            fieldID = self.ParentTFile.ParentSession.XVector[seshF[0]]
            # we separate all data by field into it's own directory called:
            # note: XVector defines this!
            fieldDir = 'Exp'+fieldID
            
            # join the root analysis path with the field specific directory:
            outDirPath = os.path.join(analPath,fieldDir)
                      
            # make the directory if it doesn't exist yet
            if not os.path.exists(outDirPath):
                os.mkdir(outDirPath)
            
            # make the filename you will use to save, adding the required tags
            tpath = self.ParentTFile.TPath
            strippedName = genF.stripTags(os.path.split(tpath)[1])
            sessionN = pSesh.SessionN
            sessionTag = '_s' + str(sessionN).zfill(4)
            timeInSeshI = self.ParentTFile.SeshT[self.FileT[0]]
            timeTag = '_t' + str(timeInSeshI).zfill(4)
            tags = sessionTag + timeTag
            outName = strippedName + tags + '.tif'

            # put the final path together
            outPath = os.path.join(outDirPath,outName)
            
            # if this path already exists we better have set overWrite to True
            if os.path.exists(outPath) and not overWrite:
                errMess = 'The directory that you are trying to save '\
                        'results to already exists and you haven\'t set '\
                        'overwrite=True. All processing is stopping.'
                raise Exception(errMess)     
            
            # now we make the sesh variable to save with with metadata and to
            # create the TFile with
            seshT = [self.ParentTFile.SeshT[T] for T in self.FileT]
            # seshF was the 1st thing defined in the loop, it's done already!
            # since we've never seen Andor split files into _z000n or _c000n,
            # we assume for now that the SeshZ and SeshC are the same as 
            # the FileZ and FileC
            # we should have also done redivideMFiles so self.FileM should 
            # correspond to SeshM
            seshM = self.FileM
            seshZ = self.FileZ
            seshC = self.FileC
            
            # this is the metadata to add:
            # you don't add channels, slices or frames b/c it does 
            # it automatically from the array shape!
            # this stuff goes into what imagej calls 'Image Description:', it
            # reads this when opening, they control display parameters
            # if you want to find what words to use to save a new parameter
            # change it in imagej and save and reopen to check imagej 
            # really does remember that stuff
            # then turn on imagej debug mode (Edit -> Options -> Misc)...
            # ... and open that image again: the parameter you changed should
            # appear in Image Description somewhere
            # this doesn't get everything! some things are stored in binary 
            #(see ijmeta below)... 
            # these are seen in debug mode as memory locations
            # e.g. to get the display ranges for each channel I: 
            # changed file in imagej, saved new file and opened in 
            # python with tifffile, imagej_metadata 
            # this is also where we write our own metadata you see...
            meta = {'hyperstack':'true','mode':'composite','unit':'um',
                    'spacing':'2.56','loop':'false','min':'0.0','max':'256',
                    'tw_NT':self.NT,'tw_NF':1,'tw_NM':self.NM,
                    'tw_NZ':self.NZ,'tw_NC':self.NC,'tw_NY':self.NY,
                    'tw_NX':self.NX,'tw_SeshT':str(seshT),
                    'tw_SeshF':str(seshF),'tw_SeshM':str(seshM),
                    'tw_SeshZ':str(seshZ),'tw_SeshC':str(seshC),
                    'tw_chan':str(self.chan)}
  
            # save the file............
            # these are the metadata that the imsave function will convert to
            # binary for imagej compatibility...
            # set ranges for each channel
            ranges = [x for i in range(self.NC) for x in [0.0,65535.0]]
            # make the LUTs
            LUTs = [genF.LUTMixer(genF.LUTDic(c)) for c in self.chan]
            # package ranges and LUTs into imagej metadata dictionary 
            ijmeta = {'Ranges':tuple(ranges),'LUTs': LUTs}
            
            # have to reshape into this shape for image j:
            dims = (self.NT*1*self.NM,self.NZ,self.NC,self.NY,self.NX)
            
            # do the save, reshaping the array for image j at the last moment
            tifffile.imsave(outPath,self.data[:,i].reshape(dims),imagej=True,
                            metadata=meta,ijmetadata=ijmeta)
            
            # update the parent xfold's record
            pXFold.SavedFilePaths.append(outPath)
            
            # now make the TFile associated with this file you just saved
            # we made the sesh variables earlier
            # create the TFile
            tfile = TFile(pSesh,None,outPath,seshT,seshF,seshM,seshZ,seshC)
            # append it to the list we are growing
            allTFiles.append(tfile)
            # add it to the session's ProcessedTFilesDic
            pSesh.ProcessedTFilesDic[analPath].append(tfile)
        
        return allTFiles
    
    
    
    def SwapXYZ(self,axisA,axisB):
        """ This can be used for swapping the XYZ axes around
        """
        # don't allow swapping of anything except XYZ for now:
        permittedAxes = [3,5,6]
        if axisA not in permittedAxes or axisB not in permittedAxes:
            errMess = 'You can only use SwapXYZ() to swap axes'\
                        '{permittedAxes}.'
            raise Exception(errMess.format(permittedAxes=permittedAxes))
        
        # do the swap
        self.data = np.swapaxes(self.data,axisA,axisB)
        
        # update other attributes
        self.updateDimensions()
        if axisA == 3 or axisB ==3:
            self.FileZ = [None for z in range(self.NZ)]
        
    
    
    def TakeXYSection(self,Xi,Xf,Yi,Yf):
        """ This takes a basic rectangular section in XY of your data. 
        Give the values you would use for a numpy array.
        """
        a = abs(Xi)>self.NX
        b = abs(Xf)>self.NX
        c = abs(Yi)>self.NY
        d = abs(Yf)>self.NY
        if any([a,b,c,d]):
            errMess = 'The values you gave for the limits in TakeXYSection()'\
            ' were out of the range of your data array.'
            raise Exception(errMess)
        self.data = self.data[:,:,:,:,:,Yi:Yf,Xi:Xf].copy()
        self.updateDimensions()
        
        
    
    def MeasureFromMask(self,outLinePathDic,dirName,manualScale=None):
        """ This measures pixel intensities from the data, with the possibility 
        of masking.
        """
        
        # make paths for saving data
        xPath = self.ParentTFile.ParentSession.ParentXFold.XPath
        measDirPath = os.path.join(os.path.split(xPath)[0],dirName)
        
        # make the directory for the csvs if it doesn't exist already
        if not os.path.exists(measDirPath):
            os.mkdir(measDirPath)  
            
        # get the data and save for each field and time
        for iF,f in enumerate(self.FileF):
            # get the fieldID and relevant outLinePath:
            seshF = self.ParentTFile.SeshF[f]
            fieldID = self.ParentTFile.ParentSession.XVector[seshF]
            outLinePath = outLinePathDic[fieldID]
            
            # make the mask from path to outLine tif:
            mask = genF.maskFromOutlinePath(outLinePath)
            
            # if your data has been downscaled you need to downscale the mask:
            if manualScale:
                if isinstance(manualScale,int):
                    manualScale = tuple([manualScale,manualScale])
                if isinstance(manualScale,list):
                    manualScale = tuple(manualScale)
                mask = downscale_local_mean(mask,manualScale)
                # re-binarize:
                mask[mask>=0.5] = 1
                mask[mask<0.5] = 0
            
            # pad the mask the same way that alignExtract would have padded it
            maskY,maskX = mask.shape
            pY = self.NY - maskY
            pad = ((pY//2,math.ceil(pY/2)),(0,0))
            mask = np.pad(mask,pad)
            pX = self.NX - maskX
            pad = ((0,0),(pX//2,math.ceil(pX/2)))
            mask = np.pad(mask,pad)
            
            for iT,t in enumerate(self.FileT):
                # start the data holder
                csvData = []
                # make the headers
                csvHeader = ['y','NX']
                for c in range(self.NC):
                    csvHeader.append(str(c))
                csvData.append(csvHeader)
                
                
                for y in range(self.NY):
                    # start building the row
                    row = [y]
                    
                    # find NX from the mask:
                    NX = np.sum(mask[y])*self.NZ
                    row.append(NX)
                    
                    for c in range(self.NC):
                        # data holder for this channel
                        cData = []
                        # collect masked data from all slices
                        for iZ in range(self.NZ):
                            czData = self.data[iF,iT,0,iZ,c,y,mask[y]]
                            cData.extend(czData)
                        # add the mean of the data for this channel
                        if NX==0:
                            row.append(0)
                        else:
                            row.append(np.mean(cData))
                    # add the new row to the data
                    csvData.append(row)
            
                # make the filename you will use to save...
                # adding the required tags
                pSesh = self.ParentTFile.ParentSession
                tpath = self.ParentTFile.TPath
                strippedName = genF.stripTags(os.path.split(tpath)[1])
                sessionN = pSesh.SessionN
                sessionTag = '_s' + str(sessionN).zfill(4)
                timeInSeshI = self.ParentTFile.SeshT[t]
                timeTag = '_t' + str(timeInSeshI).zfill(4)
                tags = sessionTag + timeTag
                csvName = strippedName + tags + '.csv'
                fieldDirName = 'Exp'+fieldID
                fieldDirPath = os.path.join(measDirPath,fieldDirName)
                # make the directory for the csvs if it doesn't exist already
                if not os.path.exists(fieldDirPath):
                    os.mkdir(fieldDirPath)                  
                # the final full path:
                csvPath = os.path.join(fieldDirPath,csvName)
                
                # save the data
                with open(csvPath, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(csvData)
                    
                    
def ConcatenateFiles(tifFileList,lim=2000000000,
                     CancelIntensityJumps=False,threshDiff=0.25):
    """ This function concatenates image data along the time axis.
        You give it either a list of tiff file paths, or a list of TFiles.
        The list of TFiles could be from an XFold's processedDataDirs.
        
        If there are tifffiles in different folders, it concatenates 
        each folder separately.
        
        It saves the concatenated tifs in a new folder...
        This folder is in the same folder as the folder that all the tiff 
        files are in. I.e. in the parent folder of the highest level folder 
        that is common to all files in the list.
        Inside that folder the folder structure of the original is recreated
        and concatenated files are saved in the corresponding places.
        The root folder of all these saved concatenated files has a name like 
        original but with the xSig taken off and concatenated added.
        
        It can also normalise intensity between time points if there are big 
        jumps in intensity. It does pairwise comparisons of the summed
        intensity of adjacent time points. This is intended for cases where 
        you have e.g. accidentally changed the laser intensity between 
        imaging sessions. So it doesn't correct small changes in summed 
        intensity but if it is bigger than the fractional threshold
        threshDiff then it will change the later time point to the total 
        intensity of the earlier.
        
        It also returns TFiles representing those saved files.
    """
    
    xSig = XFold.OutDirSig

    # extract file paths
    # if it's a string then we assume it's a path to a directory
    if isinstance(tifFileList,str):
        if os.path.exists(tifFileList):
            w = os.walk(tifFileList)
            tifFileList = [os.path.join(p,t) for (p,d,ts) in w for t in ts]
        else:
            errMess = 'You cn a string to ConcatenateFiles which we '\
                    'interpret as a path to a folder with files in. But the '\
                    'string you provided didn\'t correspond to a directory '\
                    'that exists so it has failed.'
            raise Exception(errMess)
    if all([isinstance(e,TFile) for e in tifFileList]):
        allFilePaths = [TF.TPath for TF in tifFileList]
    elif all([isinstance(e,str) for e in tifFileList]):
        allFilePaths = tifFileList
    else:
        errMess = 'Concatenate files encountered a problem in the format '\
                'of the tifFileList you provided. They must all be paths '\
                '(strings) or all TFiles.'
        raise Exception(errMess)
    
    # now we gather the file list by which directory they are in
    # do the same for the original user provided list 
    # (which may be tiff paths of TFiles)
    allDirPs = set([os.path.split(tp)[0] for tp in allFilePaths])
    allFilePaths2 = []
    tifFileList2 = []
    for i,dp in enumerate(allDirPs):
        allFilePaths2.append([])
        tifFileList2.append([])
        for j,tp in enumerate(allFilePaths):
            if os.path.split(tp)[0]==dp:
                allFilePaths2[i].append(tp)
                tifFileList2[i].append(tifFileList[j])
    allFilePaths = allFilePaths2
    tifFileList = tifFileList2
    
    # we now find 'common path'
    # this is the highest level folder that is common to all files
    commonPath = os.path.commonpath(allDirPs)
    lenComPath = len(commonPath.split(os.sep))
    # but if we can find a directory at a lower level that has xSig 
    # in it then we use that one instead... b/c xSig means that was the output
    # of some XFold analysis so it's good save at that level rather than 
    # deeper in the folder.
    if xSig in commonPath:
        for i in range(lenComPath):
            lastDir = os.path.split(commonPath)[1]
            commonPath2 = os.path.split(commonPath)[0]
            if xSig in lastDir:
                break
            else:
                commonPath = commonPath2
                lenComPath = len(commonPath.split(os.sep))
            
    # make outDirName: the common path dir name but remove xSig and add Concat_
    outDirName = os.path.split(commonPath)[1].replace(xSig,'')
    outDirName = 'Concat_'+outDirName
    # out path up to the common root folder
    outDirPath = os.path.join(os.path.split(commonPath)[0],outDirName)
    
    outTFiles = []
    # cycle through each sub directory in the original list
    for i,fpList in enumerate(allFilePaths):
        
        # get appropriate directory for this file (make it if needed)
        outName = genF.stripTags(os.path.split(fpList[0])[1])
        outPath = os.path.join(*fpList[0].split(os.sep)[lenComPath:-1]+[''])
        outPath = os.path.join(outDirPath,outPath)
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        outPath = os.path.join(outPath,outName)+'.tif'
        
        # make the directory if it doesn't exist yet
        if os.path.exists(outPath):
            errMess = 'You already seem to have concatenated files. Delete '\
                        '(or change the name of) {dirPath} and try again.'
            raise Exception(errMess.format(dirPath=outPath))
        
        # first check the files all together aren't bigger than lim
        # the default lim is 2GB but you can change it
        totalSize = []
        for tp in fpList:
            totalSize.append(os.stat(tp).st_size)
        if sum(totalSize)>lim:
            errMess = 'Total size of files is bigger than the limit of'\
            ' {lim} bytes. Concatenation was not made.'
            raise Exception(errMess.format(lim=lim))
        
        # now collect all data in a list
        dims = []
        # do first one separately, rest will be added
        with tifffile.TiffFile(fpList[0]) as tif:
            allData = tif.asarray()
            dim = genF.tif2Dims(tif)
            allData = allData.reshape(tuple(dim))
            dims.append(dim)
        # concatenate the rest:
        if len(fpList)>1:
            for tp in fpList[1:]:
                with tifffile.TiffFile(tp) as tif:
                    _data = tif.asarray()
                    dim = genF.tif2Dims(tif)
                    if dim[1:]!=dims[0][1:]:
                        errMess = 'ConcatenateFiles failed because some of '\
                            'the files had different dimensions. (size of '\
                            'time dimension can change but not the others)'
                        raise Exception(errMess)
                    _data = _data.reshape(tuple(dim))
                    allData = np.concatenate((allData,_data))
                    dims.append(dim)
        
        if CancelIntensityJumps:
            nt = sum([d[0] for d in dims])
            nc = dims[0][4]
            totals = [np.mean(allData[0,:,:,:,c]) for c in range(nc)]
            for t in range(nt-1):
                for c in range(nc):
                    tot = np.mean(allData[t+1,:,:,:,c])
                    if tot==0 or totals[c]==0:
                        totals[c] = tot
                    else:
                        frac = tot / totals[c]
                        if frac > 1 + threshDiff or frac < 1 - threshDiff:
                            allData[t+1,:,:,:,c] = allData[t+1,:,:,:,c]/frac
                        else:
                            totals[c] = tot
                        
        # shape of the final array 
        # need to put all TFM dimensions in the first axis
        nt = sum([d[0] for d in dims])
        nf = dims[0][1]
        nm = dims[0][2]
        nz = dims[0][3]
        nc = dims[0][4]
        ny = dims[0][5]
        nx = dims[0][6]
        allData = allData.reshape((nt*nf*nm,nz,nc,ny,nx))
        
        # try to find channel names in the file...
        # try to find from tw_metadata
        chanList = []
        for tp in fpList:
            with tifffile.TiffFile(tp) as tif:
                d = dir(tif)                 
                # for image j ready files that we saved:
                if ('imagej_metadata' in d and 
                        tif.imagej_metadata != None and 
                        'tw_chan' in tif.imagej_metadata.keys()):
                    chanList.append(tif.imagej_metadata['tw_chan'])
        
        # try to find from TFile data       
        chanList2 = []
        for TF in tifFileList[i]:
            if isinstance(TF,TFile):
                meta = TF.ParentSession.Metadata 
                chanList2.append(re.findall(Session.chanReg,meta))
        
        # set chan according to those results:
        if len(chanList)==len(fpList) and len(set(chanList))==1:
            chan = genF.listStr2List(chanList[0])
        elif (len(chanList)==[] and
              len(chanList2)==len(tifFileList[i]) and
              len(set(chanList2))==1):
            chan = chanList2[0]
        else:
            errMess = 'Concatenate() failed either because the information on '\
                    'channel names couldn\'t be found or because not all '\
                    'the files had the same channel names. Channel name '\
                    'information is only understood here if it is stored in the '\
                    'metadata that we make (tw_chan) or in the TFile metadata. '
            raise Exception(errMess)
        
        # set all these for the file metadata dn the TFile:
        # note how useless this TFile is going to be
        SeshT = 'Multiple sessions'
        SeshF = 'Multiple sessions'
        SeshM = 'Multiple sessions'
        SeshZ = 'Multiple sessions'
        SeshC = 'Multiple sessions'
        ParentSession = 'Multiple sessions'
        
        meta = {'hyperstack':'true','mode':'composite','unit':'um',
                'spacing':'2.56','loop':'false','min':'0.0','max':'256',
                'tw_NT':nt,'tw_NF':nf,'tw_NM':nm,
                'tw_NZ':nz,'tw_NC':nc,'tw_NY':ny,
                'tw_NX':nx,'tw_SeshT':'Multiple sessions',
                'tw_SeshF':'Multiple sessions','tw_SeshM':'Multiple sessions',
                'tw_SeshZ':'Multiple sessions','tw_SeshC':'Multiple sessions',
                'tw_chan':str(chan)}
    
        ranges = [x for i in range(nc) for x in [0.0,65535.0]]
        # make the LUTs
        LUTs = [genF.LUTMixer(genF.LUTDic(c)) for c in chan]
        # package ranges and LUTs into imagej metadata dictionary 
        ijmeta = {'Ranges':tuple(ranges),'LUTs': LUTs}
        
        # do the save, reshaping the array for image j at the last moment
        tifffile.imsave(outPath,allData,imagej=True,
                        metadata=meta,ijmetadata=ijmeta)
        
        # basically just making a blank TFile
        TFileN = None
        outTFiles.append(TFile(ParentSession,TFileN,outPath,
                         SeshT,SeshF,SeshM,SeshZ,SeshC))
    
    return outTFiles