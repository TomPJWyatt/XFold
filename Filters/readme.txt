the correction filter for each channels are stored in a tif file. the filter values should be [0 1] 
but in the tiff file everything is multiplied by 1000 because tiff takes only uint16.

background correction:

corrected_image = (raw_image -background_image)/illumination_correction_filter

the background image should be one acquired with same imaging paramters as raw_image

division is pixel by pixel
the illumination_correction_filter should be independant of imaging paramters