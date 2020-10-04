import cv2
import os
import re
import datetime
import numpy as np
import math

from . import generalFunctions as genF



def moment2Str(moment,roundM=30,roundMQ=True):
    """This takes a datetime object and converts it to a string. 
    You can do rounding in minutes by putting roundMQ=False and 
    roundM to the minutes you want to round to.
    """
    totsec = moment.total_seconds()
    h = totsec // 3600
    m = (totsec%3600) // 60
    if roundMQ:
        m = math.floor(m/roundM)*roundM
    momentString = str(int(h)).zfill(2)+':'+str(int(m)).zfill(2)
        
    return momentString



def addTimeLabel(im,time,labelScale):
    """This uses opencv to add a string 'time' to image 'im' in the bottom
        right-hand corner.
        It assumes your image is uint16 and sets the colour
        to the maximum. 
        Labelscale is how manys times bigger the ysize is 
        compared to the text height, 10 is usually good.
    """
    # set and get parameters:
    ysize,xsize = np.shape(im)
    # fontFace 
    fF = 2
    # fontScale 
    fS = 4
    # theColor 
    c = 65535
    # lineType
    lT = 8
    # labelScale
    lS = labelScale
    # thickness
    th = 4
    
    # get a text size
    textSize = cv2.getTextSize(time,fontFace=fF,fontScale=fS,thickness=th)
    
    # update the fontScale
    fS = fS*((ysize/lS)/textSize[0][1])
    
    # update the text size
    textSize = cv2.getTextSize(time,fontFace=fF,fontScale=fS,thickness=th)
    
    # set the position of the text to be kind of bottom right
    xorg = int(xsize - textSize[0][0] - xsize/200)
    yorg = int(ysize - textSize[1])
    
    # new thickness
    th = int(textSize[0][1]/10)
    
    # put the text on the image (note how this uses im being mutable)
    cv2.putText(im,time,org=(xorg,yorg),fontFace=fF,fontScale=fS,
                color=c,thickness=th,lineType=lT)
    return
    
