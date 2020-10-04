This code is for doing basic processing on microscopy data that as been obtained over multiple imaging sessions. 

'Basic processing' means montage stitching, time labelling, z-projection and several other things.
'Over multiple imaging sessions' means you may have run many different microscopy routines over the course of one experiment - this package will combine all these sessions.

The code was originally writted because we were doing experiments which lasted several days and parameters such as time interval between images, the wavelengths used and number of z-slices often varied throughout the experiment.

One of the main uses of this code is to take all this data and create a 'summary' video that looks nice and you can watch to visually understand what has happened in the experiment. 

Some other basic processing it can do are:
simple image size reduction 
deletion of blank time points (when the microscope has crashed) 
field of view homogenisation - you provide a 'filter image' of how the field of view sensitivity varies
auto levels

For more info see notes inside the code itself, especially at the beginning of Classes.py
