import sys
#sys.path.append('C:\Berkeley\DATA\cosmic_tpx3\module_file')

#import tpx3_module as tpx3
#import tpx3_module_2 as tpx3_2

import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np


#from matplotlib.ticker import MaxNLocator
#from matplotlib.colors import LogNorm

#import matplotlib.patches as mp
#from matplotlib import cm


import cv2
from skimage.util import img_as_float, img_as_ubyte


from skbeam.core import correlation as corr
#import skbeam
import skbeam.core.roi as roi
#import pycorrelate as pyc

#import struct as struct
import pandas as pd 
#from astropy.io import fits
#import sys
import re
import os
from pandas.plotting import autocorrelation_plot
import multipletau
from tqdm.notebook import trange, tqdm
from skimage import measure, color, io
from skimage.util import img_as_float, img_as_ubyte
import skbeam.core.utils as utils

from scipy import ndimage
#import matplotlib.patches as patches
from PIL import Image
from scipy.stats import t
from statsmodels.tsa.stattools import adfuller





from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS, TapTool, Slider, Div
from bokeh.plotting import figure, show #Figure
from bokeh.palettes import Magma, Inferno, Plasma, Viridis256, Cividis

from bokeh.models import Label, LabelSet, Range1d, Span





def imager_mask_remover(inputdf, contrast0, contrast1):
    """
    After reading data input the dataframe with '1did' and 't'. It is going to reconstruct
    the image in 512x512 frame. Going to return two dataframes, the first one
    has all the info, pixels, PhNo, timestamps, x, y. Timestamps are grouped by their 1did. In this
    Way memory is saved. The other one is in the image format, so that it can be used 
    to draw images, ROIs, etc. 
    """
    X4 = inputdf.groupby('1did').apply(lambda x: x['t'].values.tolist()).reset_index(name = 'ts')
    X4['PhNo'] = X4['ts'].apply(lambda x: len(x))
    
    ####Create a blank canvas
    full_1d_ids = []
    for x in range(512):
        for y in range(512):
            tempid = y*512 + x
            full_1d_ids.append(tempid)
    ### create zeros for all 100 pixels
    zero_photon_nos = np.zeros(len(full_1d_ids))
    #Df with 100 pixel ids and phohton no which is zero
    blank_canvas = pd.DataFrame({'1did': full_1d_ids, 'PhNo': zero_photon_nos})
    #### You need to find the pixels that recieved at least one photon. So only the unique 
    #### 1did is sufficient. It saves memory. 
    Xunique = inputdf.drop_duplicates(subset = ['1did'], keep = 'first')
    #############################################
    ### Find the elements in canvasdf (the 10x10 big matrix) that are not inside Xunique (
    # (the pixels that have photon counts ). The result would be pixels that never recieved photons.
    #start_time = time.process_time()
    #pixels_with_zeroPh = list(set(blank_canvas2['1did'].values) - set(X['1did'].values ))
    pixels_with_zeroPh = list(set(blank_canvas['1did'].values) - set(Xunique['1did'].values ))
    #print(time.process_time()-start_time, 'seconds')
    fill_with_zero = np.zeros(len(pixels_with_zeroPh))
    ### Create a blank dataframe that has no photon counts at any  pixel
    blank_df_allZero = pd.DataFrame({'1did': list(pixels_with_zeroPh), 'PhNo': fill_with_zero })
    
    ######## This is the df that has all info
    imagedf = pd.concat([X4[['1did', 'PhNo','ts']], blank_df_allZero], ignore_index= True)
    imagedf['x'] = imagedf['1did']%512
    imagedf['y'] = imagedf['1did']//512

    ##### This one is for imaging
    imagedf6 = imagedf[['x','y','PhNo']].pivot(index = 'x', columns = 'y', values = 'PhNo').fillna(0)
    
    plt.figure()
    plt.imshow(imagedf6, vmin = contrast0,  vmax = contrast1)

    return imagedf, imagedf6


def q_roi_viewer_new(imagefile, center_beam, distance, rad_ini, rad_fin, agrid_ini, agrid_fin,
                 vm1, vm2):

    center = center_beam #(201, 213)   #(300,223)  
    binning = 1
    pixel_size = (binning*55.0e-6, binning*55.0e-6)
    sample_distance = 1 #1.0 #m
    wavelength = (1240/706) #1e-9
    shape = (512,512) # imagefile.shape #(512,512)  #Img[0:512,0:512].shape #data[0].shape

    rad_grid = utils.radial_grid(center=center, shape=shape, pixel_size=pixel_size)
    twotheta_grid = utils.radius_to_twotheta(dist_sample=sample_distance, radius=rad_grid)
    q_grid = utils.twotheta_to_q(two_theta=twotheta_grid, wavelength=wavelength)
    angle_grid = utils.angle_grid(center=center, shape=shape, pixel_size=pixel_size)

    #df1 = masktest[0].copy(deep = True)

    df_Img = imagefile #masktest[1] #pd.DataFrame(Img[0:512, 0:512]) # Just the image. Here row and column
    # indices are nothing but the pixel cooridiantes, i.e., x = (0 -511) and y = (0-511)
    #df_q = pd.DataFrame(q_grid)  # converts the xy pixel coordinates into qx qy
    df_rad = pd.DataFrame(rad_grid) # Radial distance from the center
    #df_2theta = pd.DataFrame(twotheta_grid) # Two theta
    df_angle = pd.DataFrame(angle_grid)  # Angle 
    #Don't need all that info
    #tgf = [df_Img, df_q, df_rad, df_2theta, df_angle]
    #full = pd.concat(tgf, axis = 1, keys = ['Pixel', 'qgrid', 'rad', '2theta', 'agrid'],  ignore_index= False)
    tgf = [df_Img, df_rad, df_angle]
    full = pd.concat(tgf, axis = 1, keys = ['Pixel', 'rad', 'agrid'],  ignore_index= False)

    q_ring_df = full['Pixel'][ (full['rad'] > rad_ini) & (full['rad'] < rad_fin)
                       & (full['agrid'] > agrid_ini) & (full['agrid'] < agrid_fin)]

    plt.figure()
    plt.imshow(q_ring_df, vmin = 0, vmax = vm1, alpha = 1)
    plt.imshow(df_Img, vmin = 0, vmax = vm2,  cmap = 'viridis', alpha = 0.6)
    #plt.title("{}{}{}{}{}{}{}".format(rad_ini,'_',rad_fin, '_', agrid_ini,'_', agrid_fin))
    plt.title(f"radial limit:[{rad_ini}-- {rad_fin}]; phi limit: [{agrid_ini}--{agrid_fin}]")




def timestamps_from_ring_movie(datafile, imagefile, vm1, vm2,   center_beam, distance, 
                             rad_ini, rad_fin, agrid_ini, agrid_fin, 
                ):
    """
    datafile = is the actual dataframe that has all the x,y, and timestamp
    imagefile = just the image to show ROI, qring etc.
    Selects a q ring, collects all the timestamps within that ring, and calcualtes ACF. 
    Same as timestamps_from_ring however it outputs the full dataframe instead
    of just the timestamps
    """
    center = center_beam #(201, 213)   #(300,223)  
    binning = 1
    pixel_size = (binning*55.0e-6, binning*55.0e-6)
    sample_distance = distance #1 #1.0 #m
    wavelength = (1240/706) #1e-9
    shape = (512,512) # imagefile.shape #(512,512)  #Img[0:512,0:512].shape #data[0].shape

    rad_grid = utils.radial_grid(center=center, shape=shape, pixel_size=pixel_size)
    twotheta_grid = utils.radius_to_twotheta(dist_sample=sample_distance, radius=rad_grid)
    q_grid = utils.twotheta_to_q(two_theta=twotheta_grid, wavelength=wavelength)
    angle_grid = utils.angle_grid(center=center, shape=shape, pixel_size=pixel_size)

    #df1 = masktest[0].copy(deep = True)

    df_Img = imagefile #masktest[1] #pd.DataFrame(Img[0:512, 0:512]) # Just the image. Here row and column
# indices are nothing but the pixel cooridiantes, i.e., x = (0 -511) and y = (0-511)
    #df_q = pd.DataFrame(q_grid)  # converts the xy pixel coordinates into qx qy
    df_rad = pd.DataFrame(rad_grid) # Radial distance from the center
    #df_2theta = pd.DataFrame(twotheta_grid) # Two theta
    df_angle = pd.DataFrame(angle_grid)  # Angle 
    
    tgf = [df_Img, df_rad, df_angle]
    full = pd.concat(tgf, axis = 1, keys = ['Pixel', 'rad', 'agrid'],  ignore_index= False)
    
    q_ring_df = full['Pixel'][ (full['rad'] > rad_ini) & (full['rad'] < rad_fin)
                       & (full['agrid'] > agrid_ini) & (full['agrid'] < agrid_fin)]    
    plt.figure()
    plt.imshow(q_ring_df, vmin = 0, vmax = vm1, alpha = 1)
    plt.imshow(df_Img, vmin = 0, vmax = vm2,  cmap = 'viridis', alpha = 0.6)
    #plt.title("{}{}{}{}{}{}{}".format(rad_ini,'_',rad_fin, '_', agrid_ini,'_', agrid_fin))
    plt.title(f"radial limit:[{rad_ini}-- {rad_fin}]; phi limit: [{agrid_ini}--{agrid_fin}]")
       
    df_rad_1d = df_rad.stack().reset_index()
    df_rad_1d.columns = ['xrad', 'yrad', 'radval']
    df_rad_1d['id_rad_1d'] = df_rad_1d['yrad']*512 + df_rad_1d['xrad']

    df_angle_1d = df_angle.stack().reset_index()
    df_angle_1d.columns = ['xangle', 'yangle', 'angleval']
    df_angle_1d['id_angle_1d'] = df_angle_1d['yangle']*512 + df_angle_1d['xangle']

    
    df1 = datafile[['1did', 'ts']].dropna()  #df1 = datafile.copy(deep = True)
    
    ## We only need to find which 1did is common between the ROI dataframe and the original
    ### full dataframe. From the common dataframe we can extract all the timestamps
    df_rad_1d_sorted = df_rad_1d[['id_rad_1d', 'radval']].sort_values(by = ['id_rad_1d'], ignore_index = True)
    df_angle_1d_sorted = df_angle_1d[['id_angle_1d', 'angleval']].sort_values(by = ['id_angle_1d'], ignore_index = True)
    
    #combining the radial and angular dataframes with only their 1DID (pixel coordinates) and corre
    ## corresponding radial q and angualr phi values
    df_rad_angle_comb = df_rad_1d_sorted[['id_rad_1d', 'radval']].merge(
                                            df_angle_1d_sorted[['id_angle_1d', 'angleval']],
                                              left_on = 'id_rad_1d', right_on = 'id_angle_1d')
    ## Now based on the RoI condition, select the 1DIDs that fall within the condition.
    ## Note id_rad_1D and id_angle_1D are the same thing. Pixel coords in 1D
    df_rad_angle_ring = df_rad_angle_comb[ (df_rad_angle_comb['radval'] > rad_ini) & (df_rad_angle_comb['radval'] < rad_fin) &\
    (df_rad_angle_comb['angleval'] > agrid_ini) &(df_rad_angle_comb['angleval'] < agrid_fin)]
    ## We only need to know the 1DID, so only taking that info and renaming it to '1did' so that
    ## we can compare it to the full dataframe with timestamps.
    comb = pd.DataFrame(df_rad_angle_ring['id_rad_1d']).rename(columns = {'id_rad_1d':'1did'})
    ##### Now we find entries in the full dataframe that has the IDs identified in the
    ####### previous df (comb). This IDs (1D pixel coordinates) fall within the selected ROI
    ### This merging of two dataframes take only the common element
    new_ring_roi1 = df1.merge(comb, on = '1did', how = 'inner')   
    ### Extract all the timestamps. The timestamps are grouped by their 1DIds so you need to
    ### explode
    
    new_ts = new_ring_roi1['ts'].explode()
   
    

    return new_ts, new_ring_roi1




def full_image_for_speckle_detection(image_in):
    """
    input mask_remover[1] (image array), outputs normalized array that can be used as an input to 
    the speckle_detection scheme.
    """
    img_array = image_in.to_numpy() #mask_remover250[1].to_numpy()
    normalized_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min())) * 256
# Convert to integer type if needed
    normalized_array = normalized_array.astype(np.uint8)
    testimage = Image.fromarray(normalized_array.astype(np.uint8), 'L')
    plt.figure(figsize = (12,6))
    plt.subplot(121)
#plt.imshow(normalized_array[300:500, 0:500], cmap = 'gray')
    plt.imshow(normalized_array, cmap = 'gray')
    plt.colorbar(fraction=0.046, pad=0.08)
    plt.title('Normalized Image')
    plt.subplot(122)
    #plt.figure()
    plt.hist(normalized_array.flatten())
    plt.xlabel('Counts')
    plt.ylabel('Frequency')
    plt.tight_layout()
    return normalized_array

#normalized_array_full = full_image_for_speckle_detection(mask_remover250[1])


# In[5]:






def qring_image_for_speckle_detection(qringdf):
    """
    Input the qring dataframe, and outputs a normalized image of the qring 
    that can be used for speckle detection within the q-ring
    """
    # create a blank df (blank canvas)
    full_1d_ids = [( y*512 + x) for x in range(512) for y in range(512)]
    zero_photon_nos = np.zeros(len(full_1d_ids))
    blankdf = pd.DataFrame({'1did': full_1d_ids, 'PhNo': zero_photon_nos})
    blankdf['x'] = blankdf['1did']%512
    blankdf['y'] = blankdf['1did']//512
    #blankdf1 = blank_df.copy(deep = True)

    qdf =qringdf.copy(deep = True)
    qdf['PhNo'] = qdf['ts'].apply(lambda x: len(x))
    qdf['x'] = qdf['1did']%512
    qdf['y'] = qdf['1did']//512
        
    # find all the pixel cooridiantes that are within the qring
    q_ids =  qdf['1did'].values
    #Find the pixel coordinates in the blank canvas that are outside of the qring
    blank_minus_q = blankdf[~blankdf['1did'].isin(q_ids)]
    #print('blank, qring, subtracted shape ', blank_canvas.shape[0], qring_ts[1].shape[0], blank_canvas.shape[0]- qring_ts[1].shape[0])
    #print('blank_minus_q_df', blank_minus_q.shape[0])
    combined_df = pd.concat([qdf, blank_minus_q], ignore_index = True).fillna(0)
    qimage = combined_df[['x','y', 'PhNo']].pivot(index='x', columns='y', values='PhNo')
    img_array = qimage.to_numpy()  
    normalized_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min())) * 256
    # Convert to integer type if needed
    normalized_array = normalized_array.astype(np.uint8)
    testimage = Image.fromarray(normalized_array.astype(np.uint8), 'L')
    
    plt.figure(figsize = (16,4))
    plt.subplot(131)
    plt.imshow(normalized_array, cmap = 'gray')
    plt.colorbar(fraction=0.046, pad=0.08)
    plt.title('NormalizedImage')
    plt.subplot(132)
    plt.imshow(qimage)
    plt.title('OriginalImage')
    plt.colorbar(fraction=0.046, pad=0.08)
    plt.subplot(133)
    plt.hist(normalized_array.flatten())
    plt.xlabel('Counts')
    plt.ylabel('Frequency')
    plt.title('Count Histogram')
    plt.tight_layout()
    
    return normalized_array, qimage, qdf

    
#q1_image_norm, q1image, q1df = qring_image_for_speckle_detection(q250_1_ts[1])




def speckle_finder_cv2(input_image, kernel, sd, thresh, min_size, max_size):
# Load the image
    """
    kernel should be (5,5) in this form; positive and odd
    sd is an integer 
    thresh is an integer no
    min_size: minimum size of the detected speckle to be considered
    max_size: maximum size of the detected speckle to be considered
    """
    image = input_image #cv2.imread('scattering_image.jpg', cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, kernel, sd)
    #blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use a binary threshold to create a binary image
    _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)

    # Find contours (blobs)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter speckles based on size
    speckles = []
    areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)
        if (area > min_size) and (area <= max_size ):  # Speckles are typically small
            speckles.append(contour)

    # Draw contours on the original image
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output, speckles, -1, (0, 255, 0), 2)
    
    return output, speckles, areas


# In[ ]:





def PhotonTimeStampBinned(x, divisionNo):
    """
    This function bins the raw timestamps into user defined bin widths.
    x : photon time stamps 
    divisionNo: The number with which to divide timestamps. For instance 1e9 will create bins with 1s duration
    """
    pix1ns = np.array(x, dtype = 'int64') #* 200
    pix1sort = np.sort(pix1ns)
    divide_by = divisionNo
    num_bins = int((pix1sort[-1] - pix1sort[0])/divide_by)    #1e9 gives 1 s bin width # 1e8 gives 0.11s width
    each_bin_duration = int((pix1sort[-1] - pix1sort[0])/num_bins)/1e9
    print('No. of bins: ', num_bins, 'EachBinDuration (sec): ', each_bin_duration)
    bin_edges = np.linspace(pix1sort[0], pix1sort[-1], num_bins)
    #To remove the counts from the first and last beans, they could
    # somethimes have zero values.
    binned_data= np.digitize(pix1sort, bin_edges[1:-1]) 
    # Calculate histogram counts using bin count
    hist_binned_data = np.bincount(binned_data)
    return hist_binned_data, bin_edges



        
def ac_calculator_speckles_onlystationary(speckles_list, normalized_image_in, main_df, divide_by = 1e9, deltaT = 1, alpha_tolerance = 0.05):
    """
    We are not going to normalize the speckle signals, but will only select 
    the singals that are stationary. ACF will be done only on stationary signals.
    if pval > alpha_tolerance; non-stationary... 
    speckles_list : List of all speckles (contours); this is the output from the "identify_speckles" function
    normalized_image_in : Image normalzied from 0 to 256
    main_df: Dataframe with all data, e.g., 1did, xy coordinates, timestamps. E.g., mask_remover[0]
    divide_by: the value to divide the timestamps for binning. for 1e9 for 1s, 1e8 for 0.1s, 1e7 for 0.01s, 1e6 for 0.001s
    deltaT: each bin duration
    Outputs binned data, acfs calcualted from both MPT and Pandas Autocorrelate
    
    """
    
    #Previous version. Now we do the selection beforehand during the speckle detection phase
    #reduced_speckles = [] #select speckles that are greater than a particular area
    #for sps in speckles_list:
    #    if len(sps) > 5:
    #        reduced_speckles.append(sps)
    reduced_speckles = speckles_list
    # Since we have already chosen speckles in the previous phase; these two images are same.
    # However kept both for less hassle
    output_image_all_speckles = cv2.cvtColor(normalized_image_in.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image_all_speckles, speckles_list, -1, (0, 0, 255), 2)
   
    output_image_reduced_speckles = cv2.cvtColor(normalized_image_in.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image_reduced_speckles, reduced_speckles, -1, (0, 0, 255), 2)
    
    plt.figure()
    #plt.subplot(121)
    #plt.imshow(output_image_all_speckles) # plt.imshow(output_image)
    #plt.title('All Speckles')
    #plt.subplot(122)
    plt.imshow(output_image_reduced_speckles)
    plt.title('All Speckles')
    plt.tight_layout()
    
    # now find all the pixels that are bound by the contours.
    output_image_reduced_speckles2 = output_image_reduced_speckles.copy()
    # copied the image file where the reduced contours are drawn
    speckle_coordinates_list = []              #pts_list = []
    # For each list of contour points...
    for cnt in reduced_speckles:
    # Create a mask image that contains the contour filled in
        cimg = np.zeros_like( output_image_reduced_speckles2)
        cv2.drawContours(cimg, [cnt], 0, color=255, thickness=2) # can use -1 then less points would be there
        # thickness plays a significnat role. If thickness = 2 more pixels are going to be have the value of 255. 
        # Access the image pixels and create a 1D numpy array then add to list
        coords = np.where(cimg == 255)
        speckle_coordinates_list.append(coords)
    
    # zip xy coordinate to have format like [(x1,y1), [x2,y2], .....] and so on. Also calculate the 1did so that we can compare with dataframe.
    # Essentially, these are the speckle coordinates on which we will perform further calculations
    zipped_list_all = []
    all_1did = []
    for i in range(len(speckle_coordinates_list)):
        #(tx,ty) = (red_sp[i][:,0,1], red_sp[i][:,0,0]) 
        (tx,ty) = (speckle_coordinates_list[i][0], speckle_coordinates_list[i][1])
        zipped_list_all.append(list(zip(tx,ty)))
        tmp_1did = ty * 512 + tx
        all_1did.append(tmp_1did)
    
    tdf = main_df.copy(deep = True)   #mask_remover250[0].copy(deep = True)
    # Find the timestamps, photon counts etc. from the main dataframe for each set of speckle coordinates. Each speckle will be a dataframe that has
    # all the timestamps, coordinates etc.,. All speckle dataframes are going to be stored in a dict, e.g., spdict. For only the timestamps, we can
    # look at the tsdict. From now on, the keys in the dictionaries are going to be important because they identify the speckles and the data.
    speckles_dict = {}
    timestamps_dict = {}
    for index, ids in enumerate(all_1did):
        #print(index, ids)
        tempdf = tdf[tdf['1did'].isin(ids)].dropna()
        speckles_dict[index] = tempdf
        tempts = tempdf.explode('ts')['ts'].to_numpy().astype('int64')
        timestamps_dict[index] = tempts
    
    # Now, iterate through each df in the dictionary, that is iterate through each speckle, and bin the timestamps. 
    binned_data_edges_dict = {}
    for key, value in timestamps_dict.items():
         #print(f"Key: {key}, Value: {value}")
         tmp_bin, tmp_edge = PhotonTimeStampBinned(np.sort(value*200), divide_by)
         #normalized_tmp_bin = tmp_bin/binned_data_for_norm
         adfuller_result = adfuller(tmp_bin, autolag='AIC')
         p_val = adfuller_result[1]
         if p_val < alpha_tolerance:
            temp_bin_df = pd.DataFrame({'bins': pd.Series(tmp_bin[1:-1]), 'edges': pd.Series(tmp_edge[1:-1]) }) # discard the first and last bins as there might be error in them
            binned_data_edges_dict[key] = temp_bin_df
         elif p_val >=alpha_tolerance:
            pass
    # plot the binned data, i.e., time series    
    plt.figure()
    for key,val in binned_data_edges_dict.items():        
        plt.plot(np.cumsum(np.diff(val['edges']))/divide_by, val['bins'].dropna() )
    plt.show()
    
# calculate acf with multipletau for each binned data, that is for each speckle
    mpt_ac_dict = {}
    for key,value in binned_data_edges_dict.items():
        tmp_ac = multipletau.autocorrelate( value['bins'].dropna().astype('float64'), 
                               deltat = deltaT, normalize=True, compress = 'average')
        mpt_ac_dict[key] = tmp_ac
    # Few curves from mpt
    #plt.figure()
    #for key,val in mpt_ac_dict.items():
    #    plt.semilogx(val[:,0], val[:,1], 'o-')
    #plt.title('Plots from MPT')
    #plt.show()
    
    # Calcualte acfs with pandas autocorrelation_plot
    pandas_res_dict = {}
    
    for key,value in binned_data_edges_dict.items():
        tempax = autocorrelation_plot(value['bins'].dropna() )
        plt.xscale('log')
        plt.title('Autocorrelation (Pandas)')
        #tempax = autocorrelation_plot(value, label = "{}".format(key))
        #pandas_lags_dict[key] = tempax.lines[-1].get_xdata() 
        #pandas_acs_dict[key] = tempax.lines[-1].get_ydata() 
        temp_pandas_df = pd.DataFrame({'lag': tempax.lines[-1].get_xdata(), 'ac': tempax.lines[-1].get_ydata()    })
        pandas_res_dict[key] = temp_pandas_df
    
    #plt.figure()
    #plt.semilogx(pandas_res_dict[0]['lag'], pandas_res_dict[0]['ac'])
    #plt.semilogx(pandas_res_dict[1]['lag'], pandas_res_dict[1]['ac'])
    #[plt.axhline(y = i, linestyle = '--', color = 'gray') for i in [0.091, -0.091]]
    #[plt.axhline(y = i, linestyle = '-', color = 'gray') for i in [0.068, -0.068]]
    #plt.show()
    
    return binned_data_edges_dict, pandas_res_dict, mpt_ac_dict, speckles_dict, reduced_speckles
        





# In[11]:


def above_line_or_not(xin, yin, xmin, xmax, hline):
    """
    Determines if the acf curve is above a confidence interval line through t-testing
    xin: lag data, yin: acf
    xmin, xmax: range of the acf curve to consider for testing
    hline: 95% or 99% CI. If there are 857 bins (as in 1s binning) the value for 95% cI
    is 1.96/np.sqrt(N = 857) = 0.068. Calculate it first and then input here. Default is 0.068 
    """
    x = xin[(xin >= 0) & (xin < xmax)]
    y = yin[(xin >= 0) & (xin < xmax)]
    n = len(y)
    alpha = 0.05
    horizontal_line = hline #0.068
    y_mean = np.mean(y)
    y_std = np.std(y, ddof=1)
    t_crit = t.ppf(1 - alpha / 2, n - 1)  # Critical t-value for the confidence interval
    y_ci = t_crit * (y_std / np.sqrt(n))  # Margin of error
    # Check if the lower bound of the confidence interval is above the horizontal line
    ci_lower = y_mean - y_ci
    above_line = ci_lower > horizontal_line
    if above_line:
        return 1, y_mean, y_ci
    else:
        return 0, y_mean, y_ci
    



def dynamic_speckle_finder_3(ac_dict_in, speckles_dict_in, normalized_image_in,  all_speckle_coords, xmax, hline = 0.068):
    """
    
    """
    #chose only the acs that are above the confidence interval of white noise. Save the acs and lags into a dict
    chosen_acs_lags_pandas = {}
    chosen_confidence_interval = {}
    for key,val in ac_dict_in.items():
        xx = val['lag'].values
        yy = val['ac'].values
        res, signal_mean, signal_ci = above_line_or_not(xx, yy, 0, xmax, hline)
        if res == 1:
            tempdf = pd.DataFrame({'lag': val['lag'], 'ac': val['ac']})
            chosen_acs_lags_pandas[key] = tempdf
            ci_df = pd.DataFrame({'mean': [signal_mean], 'ci': [signal_ci], 'ci_lower': [signal_mean-signal_ci] })
            chosen_confidence_interval[key] = ci_df            
        elif res == 0:
            pass    
    
    no_of_dynamic_acs = list(chosen_acs_lags_pandas.keys())
    if len(no_of_dynamic_acs) == 0:
        print('There are No Dynamic Speckles')
        print('Calculating the average of all ACs. Returning the input data again')
        
        all_acs_pandas = [val['ac'].values for key,val in ac_dict_in.items()]
        average_all_acs_pandas = np.average(all_acs_pandas, axis = 0)
        first_key = list(ac_dict_in.keys())[0]
        #print(average_dynamic_acs_pandas)
        all_speckle_image = cv2.cvtColor(normalized_image_in, cv2.COLOR_GRAY2BGR)
        #dynamic_speckle_image = cv2.cvtColor(normalized_image_in, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(all_speckle_image, all_speckle_coords, -1, (0, 0, 255), 2)
        #cv2.drawContours(dynamic_speckle_image, dynamic_speckle_coord_aftercalc_list , -1, (0, 255, 0), 2)

        plt.figure(figsize = (12,9))
        plt.subplot(121)
        plt.imshow(all_speckle_image, vmin = 0, vmax = 100)
        plt.title('Original_From_FunctionSpeckleCoordinates')
        plt.subplot(122)
        plt.imshow(normalized_image_in, cmap = 'gray', vmin = 0, vmax = 256)
        plt.title('dynamic speckles (None)')
        plt.show()
        
        plt.figure(figsize = (12,6))
        plt.subplot(121)
        plt.semilogx(ac_dict_in[first_key]['lag'], average_all_acs_pandas)
        [plt.axhline(y = i, linestyle = '--', color = 'gray') for i in [0.091, -0.091]]
        [plt.axhline(y = i, linestyle = '-', color = 'gray') for i in [hline, -hline]]
        plt.axhline(y = 0, linestyle = '-', color = 'k')
        plt.title('Average of all ACFs')
        plt.subplot(122)
        [plt.semilogx(val['lag'], val['ac']) for key,val in ac_dict_in.items()]
        [plt.axhline(y = i, linestyle = '-', color = 'gray') for i in [hline, -hline]]
        plt.axhline(y = 0, linestyle = '-', color = 'k')
        plt.title('All ACFs')
        plt.show()
        return ac_dict_in,  speckles_dict_in, all_speckle_coords, chosen_confidence_interval, average_all_acs_pandas 
    
    elif len(no_of_dynamic_acs)>0:
        keys_to_choose = chosen_acs_lags_pandas.keys()
        #This dict will have all data, e.g., xy coordinates, timestamps, etc., for the speckles (coordiantes)
        # that show dynamic behavior. These are the dynamic speckles
        dynamic_speckle_dict = {}
        for key in keys_to_choose:
            if key in speckles_dict_in:
                dynamic_speckle_dict[key] = speckles_dict_in[key]
        print(dynamic_speckle_dict.keys())
        # Find the coordinates of the dynamic speckles so that we can plot them again.
        # Here the coordiantes will be in format like [[[295,474]], [[295,475]],....] similar to output from identify_speckle function
        #so that we can plot them later
        dynamic_speckle_coord_aftercalc = {}
        for key,val in dynamic_speckle_dict.items():
            tmp_xy_coord = val[['y', 'x']].values  
            # have to reverse xy to make it similar to the output from contour function
            # Also have to make it in the shape of (N,1,2) so that we can plot it
            tmp_xy_coord2 = tmp_xy_coord.reshape(len(tmp_xy_coord),1,2).astype(np.int32)
            dynamic_speckle_coord_aftercalc [key] = tmp_xy_coord2
    
        # Plotting all vs dynamic speckles 
        dynamic_speckle_coord_aftercalc_list = list(dynamic_speckle_coord_aftercalc.values())
    
        #Now, output both the speckles we selected, and the speckles we found to be dynamic
        all_speckle_image = cv2.cvtColor(normalized_image_in, cv2.COLOR_GRAY2BGR)
        dynamic_speckle_image = cv2.cvtColor(normalized_image_in, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(all_speckle_image, all_speckle_coords, -1, (0, 0, 255), 2)
        cv2.drawContours(dynamic_speckle_image, dynamic_speckle_coord_aftercalc_list , -1, (0, 255, 0), 2)

        plt.figure(figsize = (12,9))
        plt.subplot(121)
        plt.imshow(all_speckle_image, vmin = 0, vmax = 100)
        plt.title('Original_From_FunctionSpeckleCoordinates')
        plt.subplot(122)
        plt.imshow(dynamic_speckle_image, vmin = 0, vmax = 100)
        plt.title('dynamic speckles')
        plt.show()
    
        dynamic_acs_pandas = [val['ac'].values for key,val in chosen_acs_lags_pandas.items()]
        average_dynamic_acs_pandas = np.average(dynamic_acs_pandas, axis = 0)
        first_key = list(chosen_acs_lags_pandas.keys())[0]
        #print(average_dynamic_acs_pandas)
        plt.figure(figsize = (12,6))
        plt.subplot(121)
        plt.semilogx(chosen_acs_lags_pandas[first_key]['lag'], average_dynamic_acs_pandas)
        [plt.axhline(y = i, linestyle = '--', color = 'gray') for i in [0.091, -0.091]]
        [plt.axhline(y = i, linestyle = '-', color = 'gray') for i in [hline, -hline]]
        plt.axhline(y = 0, linestyle = '-', color = 'k')
        plt.title('Average of dynamic acs')
        plt.subplot(122)
        [plt.semilogx(val['lag'], val['ac']) for key,val in chosen_acs_lags_pandas.items()]
        [plt.axhline(y = i, linestyle = '-', color = 'gray') for i in [hline, -hline]]
        plt.axhline(y = 0, linestyle = '-', color = 'k')
        plt.title('All Dynamic ACFs')
        plt.show()
        return chosen_acs_lags_pandas,  dynamic_speckle_dict, dynamic_speckle_coord_aftercalc_list, chosen_confidence_interval, average_dynamic_acs_pandas 
    


# In[14]:


def str_to_int(string_in):
    '''
    Need to import re
    
    '''
    numbers = re.findall(r'\d+', string_in)
    integers = [int(num) for num in numbers]
    return integers



#### Detrended Fluctuation Analysis
def flatten(xss):
    return [x for xs in xss for x in xs]
#scale = 1000
#m = 1
#segments = np.floor(len(X)/scale)
#print(segments)

######### Create a function to dela with one scale. Use this function in loop for many scales
def one_segment(time_series, scale, m_order):
    segments = np.floor(len(time_series)/scale)
    X_Idx_list = []  # time series
    C_list = []
    fit_list = []
    RMS_list = []
    Index_list = []
    for i in range(int(segments)):
        Idx_start = i * scale
        Idx_stop = (i+1)* scale
        Index = np.arange(Idx_start, Idx_stop, 1)
        X_Idx = time_series[Idx_start: Idx_stop]
        C = np.polyfit(Index, X_Idx, m_order)
        fit = np.polyval(C, Index)
        RMS = np.sqrt(np.mean( (X_Idx - fit)**2 ))
        X_Idx_list.append(X_Idx)
        C_list.append(C)
        fit_list.append(fit)
        RMS_list.append(RMS)
        Index_list.append(Index)
    F = np.sqrt(np.mean( (np.array(RMS_list))**2 ))
    return Index_list, X_Idx_list, fit_list, RMS_list, F

########### calculate q dependent rms
######### Create a function to dela with one scale. Use this function in loop for many scales
def one_segment_q(time_series, scale, m_order):
    segments = np.floor(len(time_series)/scale)
    X_Idx_list = []  # time series
    C_list = []
    fit_list = []
    RMS_list = []
    Index_list = []
    for i in range(int(segments)):
        Idx_start = i * scale
        Idx_stop = (i+1)* scale
        Index = np.arange(Idx_start, Idx_stop, 1)
        X_Idx = time_series[Idx_start: Idx_stop]
        C = np.polyfit(Index, X_Idx, m_order)
        fit = np.polyval(C, Index)
        RMS = np.sqrt(np.mean( (X_Idx - fit)**2 ))
        X_Idx_list.append(X_Idx)
        C_list.append(C)
        fit_list.append(fit)
        RMS_list.append(RMS)
        Index_list.append(Index)
    #RMSq = [np.array(RMS_list)**qs[j] for j in range(len(qs))]
    #F = [np.mean(RMSq)**(1/qs[j]) for j in range(len(qs)) ]
    return Index_list, X_Idx_list, fit_list, RMS_list

#dictin = res_250K30min['q1']['q1_bin']
def dfa_one_q_iterator(dict_in, scales_in, signal_process = True, m_order = 1, fa = 0, fb = -1 ):
    F1_dict = {}
    rms_dict = {}
    for key,val in dict_in.items():
        if signal_process is True:
            Y = np.cumsum( val['bins'].dropna().values - np.mean(val['bins'].dropna().values))
        elif signal_process is False:
            Y = val['bins'].dropna().values
        F_list = []
        rms_list = []
        for i in range(len(scales_in)):
            x12, y12, fit12, rms12, F12 = one_segment(Y, scales_in[i], m_order)
            F_list.append(F12)
            rms_list.append(rms12)
        F_fit, F_fit_err = np.polyfit(np.log2(scales_in[fa:fb]), np.log2(F_list[fa:fb]), 1, cov = True)
        F_fit_y = np.polyval(F_fit, np.log2(scales_in[fa:fb]))
        final_error = np.sqrt(np.diag(F_fit_err))
        F1_dict[key] = {'F':F_list, 'F_fit_y':F_fit_y,  'slope': np.round(F_fit[0],3), 'intercept': np.round(F_fit[1],3), 
                        'slope_err': np.round(final_error[0],3), 'intercept_err': np.round(final_error[1],3)}
        rms_dict[key] = {'rms':rms_list}
    
    all_slopes = [val3['slope'] for key3,val3 in F1_dict.items()]
    all_slopes_err = [val4['slope_err'] for key4,val4 in F1_dict.items()]
    avg_slope = np.round(np.average(all_slopes),3)
    std_slope = np.round(np.std(all_slopes),3)
    error_plot_x = list(F1_dict.keys())
    
    plt.figure(figsize = (16,6))
    plt.subplot(121)
    for key1, val1 in F1_dict.items():
        plt.plot(np.log2(scales_in[fa:fb]), np.log2(val1['F'][fa:fb]), 'o', label = '{}'.format(key1))
        plt.plot(np.log2(scales_in[fa:fb]), val1['F_fit_y'], '-', lw = 3)
    plt.xlabel('log2 (scales)')
    plt.ylabel('log2 F')
    plt.subplot(122)
    plt.errorbar(error_plot_x, all_slopes,
                                yerr = all_slopes_err, fmt = 'ko', ms = 10,
            linewidth=2, capsize=8)
    plt.xlabel('Speckle Index')
    plt.ylabel('Slope w/ err')
    
    return F1_dict, rms_dict, [avg_slope, std_slope] 
            
##########################################




#img_200K = speckle_image_dict_maker(['q1', 'q2', 'q3'], res_200K, dynamic_data_200K, True )


################### From 250K calculation with various bin size 
# sometimes for smaller bin sizes, the chosen acf arrays may have different sizes
# to deal with that use tolerant_mean fucntion
def tolerant_mean(arrs):
    """
    Calcualte average of numpy arrays that are unequal along 'columns'
    """
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)



#### Plots all the ACF vs Delay curves with Bokeh so tat the curves can be accessed with a slider one by one

def bokeh_plotter(ydict_values, ydict_keys, lag, hline = 0.047):
########## BOkeh
    lines_y = ydict_values
    #lines_y = [val['ac'].values for key,val in ydict.items()]
    #lines_y = list(ydict.values())    #list(pandas_acs_dict.values())
#lags_ex = [lags for i in range(len(g2.T))]
################################
########## g2s with slider
    #keys_in_dict = list(ydict.keys())
    keys_in_dict = list(ydict_keys)
    keys = list(np.arange(1, len(lines_y)+1, 1))         #list(ydict.keys())    #list(pandas_acs_dict.keys()) #np.unique(roi_grid_inds_ac))
    keys_str = list(map(str, keys))
    g2_dict = dict(zip(keys_str, lines_y))

# g2_dict = {'1': lines_y[0],
#            '2': lines_y[1],
#            '3': lines_y[2],
#            '4': lines_y[3]}
    #x = ydict[0]['lag'].values #pandas_lags_dict[0]   #lags
    x = lag
# trigonometric_functions = {
#     '0': np.sin(x),
#     '1': np.cos(x),
#     '2': np.tan(x),
#     '3': np.arctan(x)}
    initial_function = '1' #keys_str[1]#'1'

# Wrap the data in two ColumnDataSources
    source_visible = ColumnDataSource(data=dict(
        x=x, y=g2_dict[initial_function]))
    source_available = ColumnDataSource(data=g2_dict)

# Plotting the g2's
# Define plot elements
    plotsld = figure(x_axis_type = 'log', plot_width=800, plot_height=400)
    plotsld.line('x', 'y', source=source_visible, line_width=2, line_alpha=0.6,
             legend_label="{}".format(keys[int(initial_function)])  )
#plotsld.y_range=Range1d(0.95, 1.3)

#plot the markers 
    plotsld.circle('x', 'y', source=source_visible )
#plotsld.y_range=Range1d(0.95, 1.3)
    hline95pos = Span(location= hline, dimension='width', line_color='red', 
             line_width=2, line_dash='dashed') 
    hline95neg = Span(location= -hline, dimension='width', line_color='red', 
             line_width=2, line_dash='dashed') 


    slider = Slider(title='ROI Index',
                value=int(initial_function),
                start= ( np.min([int(i) for i in g2_dict.keys()]) ),
                end=np.max([int(i) for i in g2_dict.keys()])  ,
                step=1)

# Define CustomJS callback, which updates the plot based on selected function
# by updating the source_visible ColumnDataSource.
    callback = CustomJS(
        args=dict(source_visible=source_visible,
              source_available=source_available), code="""
        var selected_function = cb_obj.value;
        // Get the data from the data sources
        var data_visible = source_visible.data;
        var data_available = source_available.data;
        // Change y-axis data according to the selected value
        data_visible.y = data_available[selected_function];
        // Update the plot
        source_visible.change.emit();
        """)

    slider.js_on_change('value', callback)

    div = Div(text= '<b>text</b>', style={'font-size': '150%', 'color': 'blue'})
#str_list = ['text0', 'text1', 'text2']
    str_list = list(map(str, keys_in_dict)) #list(map(str, keys))
    str_slider = Slider(start=0, end=len(str_list)-1, value=0, step=1, title="string")


    callback = CustomJS(args=dict(div=div, str_list = str_list, str_slider=str_slider), 
                        code="""
    const v = str_slider.value
    div.text = str_list[v]
    """)
    str_slider.js_on_change('value', callback)

#layout = row( column(plotsum, plotlab), column(plotsld, slider, str_slider, div))
    layout =  column(plotsld, slider, str_slider, div)
    plotsld.add_layout(hline95pos)
    plotsld.add_layout(hline95neg)

    show(layout)


# In[3]:



def bokeh_plotter_with_ci(ydict_values, ydict_keys, lag, ci_dict, hline):
########## BOkeh
    lines_y = ydict_values
    sig_ci = [val['ci'].values for key,val in ci_dict.items()]
    sig_mean = [val['mean'].values for key,val in ci_dict.items()]
    sig_ci_ext = [np.full(len(lines_y[i]), sig_ci[i] ) for i in range(len(lines_y))]
    sig_mean_ext = [np.full(len(lines_y[i]), sig_mean[i] ) for i in range(len(lines_y))]
    #lines_y = [val['ac'].values for key,val in ydict.items()]
    #lines_y = list(ydict.values())    #list(pandas_acs_dict.values())
#lags_ex = [lags for i in range(len(g2.T))]
################################
########## g2s with slider
    #keys_in_dict = list(ydict.keys())
    keys_in_dict = list(ydict_keys)
    keys = list(np.arange(1, len(lines_y)+1, 1))         #list(ydict.keys())    #list(pandas_acs_dict.keys()) #np.unique(roi_grid_inds_ac))
    keys_str = list(map(str, keys))
    g2_dict = dict(zip(keys_str, lines_y))
    
    upper_ci_dict = dict(zip(keys_str, np.array(sig_mean_ext) + np.array(sig_ci_ext)))
    lower_ci_dict = dict(zip(keys_str, np.array(sig_mean_ext) - np.array(sig_ci_ext) ))

    #lower_ci_dict = dict(zip(keys_str, sig_mean-sig_ci))
    
    x = lag

    initial_function = '1' #keys_str[1]#'1'

# Wrap the data in two ColumnDataSources
    source_visible = ColumnDataSource(data=dict(
        x=x, y=g2_dict[initial_function]))
    source_available = ColumnDataSource(data=g2_dict)
    
    source_visible1 = ColumnDataSource(data=dict(
        x=x, y=upper_ci_dict[initial_function]))
    source_available1 = ColumnDataSource(data=upper_ci_dict)
    
    source_visible2 = ColumnDataSource(data=dict(
        x=x, y=lower_ci_dict[initial_function]))
    source_available2 = ColumnDataSource(data=lower_ci_dict)
    
# Plotting the g2's
# Define plot elements
    plotsld = figure(x_axis_type = 'log', plot_width=800, plot_height=400)
    plotsld.line('x', 'y', source=source_visible, line_width=2, line_alpha=0.6,
             legend_label="{}".format(keys[int(initial_function)])  )
#plotsld.y_range=Range1d(0.95, 1.3)

#plot the markers 
    plotsld.circle('x', 'y', source=source_visible )
    
    plotsld.line('x', 'y', source=source_visible1, line_color = 'red', line_width=2, line_alpha=0.6, line_dash = 'dotdash',
             legend_label="{}".format(keys[int(initial_function)])  )
    plotsld.line('x', 'y', source=source_visible2, line_color = 'red', line_width=2, line_alpha=0.6, line_dash = 'dotdash',
             legend_label="{}".format(keys[int(initial_function)])  )
#plotsld.y_range=Range1d(0.95, 1.3)
    hline95pos = Span(location= hline, dimension='width', line_color='gray', 
             line_width=2, line_dash='dashed') 
    hline95neg = Span(location= -hline, dimension='width', line_color='gray', 
             line_width=2, line_dash='dashed') 


    slider = Slider(title='ROI Index',
                value=int(initial_function),
                start= ( np.min([int(i) for i in g2_dict.keys()]) ),
                end=np.max([int(i) for i in g2_dict.keys()])  ,
                step=1)

# Define CustomJS callback, which updates the plot based on selected function
# by updating the source_visible ColumnDataSource.
    callback = CustomJS(
        args=dict(source_visible= source_visible, 
              source_available=source_available,
                 source_visible1= source_visible1, 
              source_available1=source_available1,
                 source_visible2= source_visible2, 
              source_available2=source_available2), code="""
        var selected_function = cb_obj.value;
        // Get the data from the data sources
        var data_visible = source_visible.data;
        var data_available = source_available.data;
        var data_visible1 = source_visible1.data;
        var data_available1 = source_available1.data;
        var data_visible2 = source_visible2.data;
        var data_available2 = source_available2.data;
        // Change y-axis data according to the selected value
        data_visible.y = data_available[selected_function];
        data_visible1.y = data_available1[selected_function];
        data_visible2.y = data_available2[selected_function];

        // Update the plot
        source_visible.change.emit();
       
        """)

    slider.js_on_change('value', callback)

    div = Div(text= '<b>text</b>', style={'font-size': '150%', 'color': 'blue'})
#str_list = ['text0', 'text1', 'text2']
    str_list = list(map(str, keys_in_dict)) #list(map(str, keys))
    str_slider = Slider(start=0, end=len(str_list)-1, value=0, step=1, title="string")


    callback = CustomJS(args=dict(div=div, str_list = str_list, str_slider=str_slider), 
                        code="""
    const v = str_slider.value
    div.text = str_list[v]
    """)
    str_slider.js_on_change('value', callback)

#layout = row( column(plotsum, plotlab), column(plotsld, slider, str_slider, div))
    layout =  column(plotsld, slider, str_slider, div)
    plotsld.add_layout(hline95pos)
    plotsld.add_layout(hline95neg)

    show(layout)


