# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:20:14 2024

@author: AIUsSaleheen
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:02:12 2024

@author: AIUsSaleheen
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:47:47 2024

@author: AIUsSaleheen
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:18:33 2023

@author: AIUsSaleheen
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:08:14 2021

@author: AIUsSaleheen
"""

#import skbeam

from skbeam.core import correlation as corr
import skbeam
import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np


from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm

import matplotlib.patches as mp
from matplotlib import cm


import cv2
from skimage.util import img_as_float, img_as_ubyte

import skbeam.core.roi as roi
import pycorrelate as pyc

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:47:24 2021

@author: sumit
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:42:12 2021

@author: sumit
"""

import numpy as np
import struct as struct
from matplotlib import pyplot as plt
import pandas as pd 
from astropy.io import fits
import sys
import pickle as pickle
import re
import os

import skbeam.core.utils as utils
import skbeam.core
import sparse


# df_full =  pd.read_csv(curr_file, sep='\t', header = None,  engine='python',
#                                  usecols = [i for i in range(2)])


########### For explanation and file without function go to 
## C:\Berkeley\DATA\cosmic_tpx3\moduleTest\200K_roi_tr1.py


def file_read(folder_path, file_name):
    thefilepath = os.path.join(folder_path, file_name)
    filedf = pd.read_csv(thefilepath, sep = '\t', header = None, skiprows = 1)
    filedf.columns = ['1did', 't']
    return filedf



def file_read_chunk(folder_path, file_name, csize):
    thefilepath = os.path.join(folder_path, file_name)
    filedf = pd.read_csv(thefilepath, sep = '\t', header = None, skiprows = 1,
                        chunksize = csize )
    firstchunk = next(filedf)
    #filedf.columns = ['1did', 't']
    firstchunk.columns = ['1did', 't']
    return firstchunk

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



def imager_hist(inputdf, contrast1, contrast2):
    inputdf['x'] = inputdf['1did']%512
    inputdf['y'] = inputdf['1did']//512
    xedges = np.arange(0,512,1)
    yedges = np.arange(0,512,1)

    x = inputdf['x'].values
    y = inputdf['y'].values
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
# Histogram does not follow Cartesian convention (see Notes),
# therefore transpose H for visualization purposes.
    H = H.T
    plt.figure()
    plt.imshow(H, interpolation='nearest', origin='upper',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin = contrast1, vmax = contrast2)
    plt.colorbar()
    plt.tight_layout()
    return H


def imager_hist2(inputdf, contrast1, contrast2):
    #inputdf['x'] = inputdf['1did']%512
    #inputdf['y'] = inputdf['1did']//512
    xedges = np.arange(0,512,1)
    yedges = np.arange(0,512,1)

    x = inputdf['x'].values
    y = inputdf['y'].values
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
# Histogram does not follow Cartesian convention (see Notes),
# therefore transpose H for visualization purposes.
    H = H.T
    plt.figure()
    plt.imshow(H, interpolation='nearest', origin='upper',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin = contrast1, vmax = contrast2)
    plt.colorbar()
    plt.tight_layout()
    return H

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
    plt.title("{}{}{}{}{}{}{}".format(rad_ini,'_',rad_fin, '_', agrid_ini,'_', agrid_fin))




def timestamps_from_ring_new(datafile, imagefile,  center_beam, distance, 
                             rad_ini, rad_fin, agrid_ini, agrid_fin, 
                 flattened, after_extraction_plot):
    """
    datafile = is the actual dataframe that has all the x,y, and timestamp
    imagefile = just the image to show ROI, qring etc.
    Selects a q ring, collects all the timestamps within that ring, and calcualtes ACF. 
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
    plt.imshow(q_ring_df, vmin = 0, vmax = 500, alpha = 1)
    plt.imshow(df_Img, vmin = 0, vmax = 800,  cmap = 'viridis', alpha = 0.7)
    plt.title("{}{}{}{}{}{}{}".format(rad_ini,'_',rad_fin, '_', agrid_ini,'_', agrid_fin))

       
    df_rad_1d = df_rad.stack().reset_index()
    df_rad_1d.columns = ['xrad', 'yrad', 'radval']
    df_rad_1d['id_rad_1d'] = df_rad_1d['yrad']*512 + df_rad_1d['xrad']

    df_angle_1d = df_angle.stack().reset_index()
    df_angle_1d.columns = ['xangle', 'yangle', 'angleval']
    df_angle_1d['id_angle_1d'] = df_angle_1d['yangle']*512 + df_angle_1d['xangle']

    if flattened is False:
        df1 = datafile[['1did', 'ts']].dropna()  #df1 = datafile.copy(deep = True)
    elif flattened is True:
        df1 = datafile[['1did', 'flat_ts']].dropna() 
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
    df_rad_angle_ring = df_rad_angle_comb[ (df_rad_angle_comb['radval'] > rad_ini) & (df_rad_angle_comb['radval'] < rad_fin) &(df_rad_angle_comb['angleval'] > agrid_ini) & (df_rad_angle_comb['angleval'] < agrid_fin)]
    ## We only need to know the 1DID, so only taking that info and renaming it to '1did' so that
    ## we can compare it to the full dataframe with timestamps.
    comb = pd.DataFrame(df_rad_angle_ring['id_rad_1d']).rename(columns = {'id_rad_1d':'1did'})
    ##### Now we find entries in the full dataframe that has the IDs identified in the
    ####### previous df (comb). This IDs (1D pixel coordinates) fall within the selected ROI
    ### This merging of two dataframes take only the common element
    new_ring_roi1 = df1.merge(comb, on = '1did', how = 'inner')   
    ### Extract all the timestamps. The timestamps are grouped by their 1DIds so you need to
    ### explode
    if flattened is False:
        new_ts = new_ring_roi1['ts'].explode()
    elif flattened is True:
        new_ts = new_ring_roi1['flat_ts'].explode()
    
    #### Now if you wanna see the ROI selection is correct, reconstruct the ROI from the
    ### Extracted x,y coordinates and Timestamps
    if after_extraction_plot is True:
        for_image = new_ring_roi1.copy()            
        for_image['PhNo'] = for_image['ts'].apply(lambda x: len(x))

        for_image['x'] = for_image['1did']%512
        for_image['y'] = for_image['1did']//512
        #for_image2 = for_image[['x', 'y', 'PhNo']]
        for_image_plot = for_image[['x','y', 'PhNo']].pivot(index='x', columns='y', values='PhNo').fillna(0)
        imagex = for_image['x'].values
        imagey = for_image['y'].values
        imageno = for_image['PhNo'].values
        Image1 = np.zeros((512,512), dtype=np.int32)
        for k in range(len(imagex)):
            Image1[imagex[k], imagey[k]] = Image1[imagex[k], imagey[k]] +1

        plt.figure()
        plt.imshow(Image1)
        plt.imshow(df_Img, vmin = 0, vmax = 9000,  cmap = 'viridis', alpha = 0.7)
        plt.title("{}{}{}{}{}{}{}".format(rad_ini,'_',rad_fin, '_', agrid_ini,'_', agrid_fin))

        plt.figure()
        plt.imshow(for_image_plot, vmin = 0, vmax = 2000, alpha = 1, extent =[0,512,0,512], aspect = 'auto')
        plt.title('DirectlyFrom_for_image2')

    return new_ts



def timestamps_from_ring_movie(datafile, imagefile,  center_beam, distance, 
                             rad_ini, rad_fin, agrid_ini, agrid_fin, 
                 flattened, after_extraction_plot):
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
    plt.imshow(q_ring_df, vmin = 0, vmax = 500, alpha = 1)
    plt.imshow(df_Img, vmin = 0, vmax = 800,  cmap = 'viridis', alpha = 0.7)
    plt.title("{}{}{}{}{}{}{}".format(rad_ini,'_',rad_fin, '_', agrid_ini,'_', agrid_fin))

       
    df_rad_1d = df_rad.stack().reset_index()
    df_rad_1d.columns = ['xrad', 'yrad', 'radval']
    df_rad_1d['id_rad_1d'] = df_rad_1d['yrad']*512 + df_rad_1d['xrad']

    df_angle_1d = df_angle.stack().reset_index()
    df_angle_1d.columns = ['xangle', 'yangle', 'angleval']
    df_angle_1d['id_angle_1d'] = df_angle_1d['yangle']*512 + df_angle_1d['xangle']

    if flattened is False:
        df1 = datafile[['1did', 'ts']].dropna()  #df1 = datafile.copy(deep = True)
    elif flattened is True:
        df1 = datafile[['1did', 'flat_ts']].dropna() 
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
    df_rad_angle_ring = df_rad_angle_comb[ (df_rad_angle_comb['radval'] > rad_ini) & (df_rad_angle_comb['radval'] < rad_fin) &(df_rad_angle_comb['angleval'] > agrid_ini) & (df_rad_angle_comb['angleval'] < agrid_fin)]
    ## We only need to know the 1DID, so only taking that info and renaming it to '1did' so that
    ## we can compare it to the full dataframe with timestamps.
    comb = pd.DataFrame(df_rad_angle_ring['id_rad_1d']).rename(columns = {'id_rad_1d':'1did'})
    ##### Now we find entries in the full dataframe that has the IDs identified in the
    ####### previous df (comb). This IDs (1D pixel coordinates) fall within the selected ROI
    ### This merging of two dataframes take only the common element
    new_ring_roi1 = df1.merge(comb, on = '1did', how = 'inner')   
    ### Extract all the timestamps. The timestamps are grouped by their 1DIds so you need to
    ### explode
    if flattened is False:
        new_ts = new_ring_roi1['ts'].explode()
    elif flattened is True:
        new_ts = new_ring_roi1['flat_ts'].explode()
    
    #### Now if you wanna see the ROI selection is correct, reconstruct the ROI from the
    ### Extracted x,y coordinates and Timestamps
    if after_extraction_plot is True:
        for_image = new_ring_roi1.copy()            
        for_image['PhNo'] = for_image['ts'].apply(lambda x: len(x))

        for_image['x'] = for_image['1did']%512
        for_image['y'] = for_image['1did']//512
        #for_image2 = for_image[['x', 'y', 'PhNo']]
        for_image_plot = for_image[['x','y', 'PhNo']].pivot(index='x', columns='y', values='PhNo').fillna(0)
        imagex = for_image['x'].values
        imagey = for_image['y'].values
        imageno = for_image['PhNo'].values
        Image1 = np.zeros((512,512), dtype=np.int32)
        for k in range(len(imagex)):
            Image1[imagex[k], imagey[k]] = Image1[imagex[k], imagey[k]] +1

        plt.figure()
        plt.imshow(Image1)
        plt.imshow(df_Img, vmin = 0, vmax = 9000,  cmap = 'viridis', alpha = 0.7)
        plt.title("{}{}{}{}{}{}{}".format(rad_ini,'_',rad_fin, '_', agrid_ini,'_', agrid_fin))

        plt.figure()
        plt.imshow(for_image_plot, vmin = 0, vmax = 2000, alpha = 1, extent =[0,512,0,512], aspect = 'auto')
        plt.title('DirectlyFrom_for_image2')

    return new_ts, new_ring_roi1

# SVS/mvie_ring_roi_xsvs_240K_RectROI.py

def timestamp_for_rois_svs(inputdf, input_image,  roi_selection, flattened):
    """
    Select timestamps for the choice of ROIs. 
    Returns a list of timestamps corresponding to the chosen ROIs. 
    You can choose if you want to use dataframe with flattened timestamps or dataframe with
    non-flattened (as is) timestamps.
    """
#######################
#df1 =  full_image[['x', 'y', 't']].copy(deep = True)
#df1 =  full_image[['x', 'y', 'flat_ts']].copy(deep = True)
    if flattened is True:
        df1 =  inputdf[['1did','x', 'y', 'flat_ts']].copy(deep = True)
    elif flattened is False:
        df1 =  inputdf[['1did', 'x', 'y', 'ts', 'PhNo']].copy(deep = True)

    rois_all = roi_selection[:]
    #Image1 = np.zeros((512,512), dtype=np.int32)

    x_all = []
    y_all = []
    t_all = []
    result_dict = {}
    image_copy = input_image.to_numpy().copy()  #full_image6.to_numpy().copy()
    drawing_image = np.ascontiguousarray(image_copy, dtype=np.float64)    
    for m in range(len(rois_all)):
        temp_df =  df1[ (df1['x'] >= rois_all[m][1]) & (df1['x'] < rois_all[m][1]+rois_all[m][3] ) 
                   & (df1['y'] >= rois_all[m][0] ) & (df1['y'] < rois_all[m][0]+rois_all[m][2] )]
        x_all.append(temp_df['x'].tolist())
        y_all.append(temp_df['y'].tolist())
        result_dict[m] = temp_df
        if flattened is True:
            #t_all.append(temp_df['flat_ts'].tolist())
            t_all.append( list(temp_df['flat_ts'].values[~np.isnan(temp_df['flat_ts'].values)]))
        elif flattened is False:    
            #t_all.append#(temp_df['t'].tolist())
            #t_all.append( list(temp_df['ts'].values[~np.isnan(temp_df['ts'].values)]))
            t_all.append( temp_df['ts'].explode().dropna().values.astype('int64'))
      
        #roi_image = temp_df[['x','y','PhNo']].pivot(index = 'x', columns = 'y', values = 'PhNo').fillna(0)
        imagex = temp_df['x'].values
        imagey = temp_df['y'].values
        Image1 = np.zeros((512,512), dtype=np.int32)
        for k in range(len(imagex)):
            Image1[imagex[k], imagey[k]] = Image1[imagex[k], imagey[k]] +1
        
        plt.figure()
        plt.imshow(Image1) 
        plt.imshow(drawing_image, cmap = 'viridis', vmin = 0, vmax = 300, 
                       alpha = 0.3)
       
    
    # for k in range(len(x_all[0])):
    #     Image1[ int(x_all[0][k]), int(y_all[0][k])] = Image1[ int(x_all[0][k]), int(y_all[0][k])] +1

    # plt.figure()
    # plt.imshow(Image1, cmap = 'viridis', vmin = 0, vmax = 300)
    # plt.title('First ROI in grid ')
    # print('{}{}{}{}'.format('len(t_all) = ', len(t_all), ' # of ROIs ', len(rois_all)  ))
    return t_all, result_dict






def basic_ac_calc(timestamps, n_casc, n_c, divby): 
    t_all = timestamps[~np.isnan(timestamps)]
    
    #t_all = timestamps
    N_casc = n_casc #32 #32 # 10 number of cascades
    N_c =  n_c  # 8   # 16 number of correlation times (delay times) per cascade
    #t_tac = 5120 # this is the t_TAC in the paper
    k_range = N_casc * N_c   # The total number of delay times
    tau_list = []  
    floor_thing = []
########### Tau according to  the rsi paper....My original Taus
####################################################
    for k in range(0,k_range):
        if k == 0:
            tc = 1
            tau_list.append(tc)
        elif k > 0 and k < 2*N_c:
            tc = tau_list[k-1] +1
            tau_list.append(tc)
        elif k >= (2 * N_c):
            tc = tau_list[k-1] + 2**np.floor( (k/N_c) -1)
            tau_list.append(tc)
            floor_thing.append(2**np.floor( (k/N_c) -1))

    tau_array = np.array(tau_list)   # Delay times to be used

    tau = tau_array[:] * divby  # Delay times in integer multiples of 1280
  
######## New Way
    
    t_all_arr = np.array(t_all)*divby 
#tmp_y =  [np.round( np.sort(t_all_arr[n])/200, 0) for n in range(len(t_all_arr))]
    #### original
    tmp_y =  np.sort( list(map(int, t_all_arr)))
    #### testing on sept25
    #tmp_y =  np.sort( list(map(int, t_all_arr )))
    #tmp_y =  np.sort( np.round(t_all_arr)/divby, 0) 
#tmp_y =  [np.sort(t_all_arr[n]) for n in range(len(t_all_arr))]

#duration = [ max(t_all[m]) - min(t_all[m])  for m in range(len(t_all)) ]
    duration = max(tmp_y) - min(tmp_y)                                                                                                                        
    idtau = np.where( duration - tau > 0 )
    tau_mod = tau[idtau]

    G = pyc.pcorrelate(tmp_y, tmp_y, tau, normalize = False) 

    plt.figure()
    plt.plot( (tau[2:])/1e9, G[1:], 'o-')
    plt.xscale('log')
    plt.tight_layout()

    GN = pyc.pcorrelate(tmp_y, tmp_y, tau_mod, normalize = True) 
    
    plt.figure()
    plt.plot( (tau_mod[2:])/1e9, GN[1:], '-' )
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()

    return tau, G, tau_mod, GN
    


def basic_ac_calc_loop(timestamps, n_casc, n_c, divby): 
    t_all = timestamps[~np.isnan(timestamps)]
    
    #t_all = timestamps
    N_casc = n_casc #32 #32 # 10 number of cascades
    N_c =  n_c  # 8   # 16 number of correlation times (delay times) per cascade
    #t_tac = 5120 # this is the t_TAC in the paper
    k_range = N_casc * N_c   # The total number of delay times
    tau_list = []  
    floor_thing = []
########### Tau according to  the rsi paper....My original Taus
####################################################
    for k in range(0,k_range):
        if k == 0:
            tc = 1
            tau_list.append(tc)
        elif k > 0 and k < 2*N_c:
            tc = tau_list[k-1] +1
            tau_list.append(tc)
        elif k >= (2 * N_c):
            tc = tau_list[k-1] + 2**np.floor( (k/N_c) -1)
            tau_list.append(tc)
            floor_thing.append(2**np.floor( (k/N_c) -1))

    tau_array = np.array(tau_list)   # Delay times to be used

    tau = tau_array[:] * divby  # Delay times in integer multiples of 1280
  
######## New Way
    
    t_all_arr = np.array(t_all)*divby 
#tmp_y =  [np.round( np.sort(t_all_arr[n])/200, 0) for n in range(len(t_all_arr))]
    #### original
    tmp_y =  np.sort( list(map(int, t_all_arr)))
    #### testing on sept25
    #tmp_y =  np.sort( list(map(int, t_all_arr )))
    #tmp_y =  np.sort( np.round(t_all_arr)/divby, 0) 
#tmp_y =  [np.sort(t_all_arr[n]) for n in range(len(t_all_arr))]

#duration = [ max(t_all[m]) - min(t_all[m])  for m in range(len(t_all)) ]
    duration = max(tmp_y) - min(tmp_y)                                                                                                                        
    idtau = np.where( duration - tau > 0 )
    tau_mod = tau[idtau]

    G = pyc.pcorrelate(tmp_y, tmp_y, tau, normalize = False) 

    # plt.figure()
    # plt.plot( (tau[2:])/1e9, G[1:], 'o-')
    # plt.xscale('log')
    # plt.tight_layout()

    GN = pyc.pcorrelate(tmp_y, tmp_y, tau_mod, normalize = True) 
    
    # plt.figure()
    # plt.plot( (tau_mod[2:])/1e9, GN[1:], '-' )
    # plt.xscale('log')
    # plt.legend()
    # plt.tight_layout()

    return tau, G, tau_mod, GN
    



    
def basic_ac_calc_linear(timestamps, n_casc, n_c, divby,linear_limit, 
                         linear_interval, tau_multi, 
                         tau_linear,  
                         sort_ts
                        ):
    t_all = timestamps[~np.isnan(timestamps)]
    
    if tau_multi is True:
        N_casc = n_casc #32 #32 # 10 number of cascades
        N_c =  n_c  # 8   # 16 number of correlation times (delay times) per cascade
        #t_tac = 5120 # this is the t_TAC in the paper
        k_range = N_casc * N_c   # The total number of delay times
        tau_list = []  
        floor_thing = []
        ########### Tau according to  the rsi paper....My original Taus
        ####################################################
        for k in range(0,k_range):
            if k == 0:
                tc = 1
                tau_list.append(tc)
            elif k > 0 and k < 2*N_c:
                tc = tau_list[k-1] +1
                tau_list.append(tc)
            elif k >= (2 * N_c):
                tc = tau_list[k-1] + 2**np.floor( (k/N_c) -1)
                tau_list.append(tc)
                floor_thing.append(2**np.floor( (k/N_c) -1))

        tau_array = np.array(tau_list)   # Delay times to be used
        tau = tau_array[:] * 1  # Delay times in integer multiples of 1280
        
    elif tau_linear is True:
        tau_array = np.arange(1,linear_limit,linear_interval)  #19000
        tau = tau_array[:]*1  #######

    t_all_arr = np.array(t_all)*divby 
#tmp_y =  [np.round( np.sort(t_all_arr[n])/200, 0) for n in range(len(t_all_arr))]
    tmp_y =  np.sort( list(map(int, t_all_arr/divby )))
    #tmp_y =  np.sort( np.round(t_all_arr)/divby, 0) 
#tmp_y =  [np.sort(t_all_arr[n]) for n in range(len(t_all_arr))]

#duration = [ max(t_all[m]) - min(t_all[m])  for m in range(len(t_all)) ]
    duration = max(tmp_y) - min(tmp_y)                                                                                                                        
    idtau = np.where( duration - tau > 0 )
    tau_mod = tau[idtau]

    G = pyc.pcorrelate(tmp_y, tmp_y, tau, normalize = False) 

    plt.figure()
    plt.plot( (tau[2:]*divby)/1e9, G[1:], 'o-')
    plt.xscale('log')
    plt.tight_layout()

    GN = pyc.pcorrelate(tmp_y, tmp_y, tau_mod, normalize = True) 
    
    plt.figure()
    plt.plot( (tau_mod[2:]*divby)/1e9, GN[1:], '-' )
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()

    return tau, G, tau_mod, GN    





def roi_creator(roi_list,  input_image, contr_alpha, contr_beta):

    """
    Creates a list of square or rectangular rois.
    input list like this [[x1,y1, dx1, dy1], [x2,y2,dx2,dy2],.....]
    x and y as in the matplotlib plot
    """   
    
# coordinates of the upper-left corner and width and height of each rectangle
    roi_grid_ac = np.array((roi_list),
                        dtype=np.int64)

    shape = input_image.shape  #(512,512)  #full_image.shape #(512,512)  # data.shape[1:]
# to the order they are specified in coords.
    label_array_grid_ac = roi.rectangles(roi_grid_ac, shape)
    roi_grid_inds_ac, pixel_grid_list_ac = roi.extract_label_indices(
    label_array_grid_ac)

########################## NOw the plotting of the ROIs

    image_copy = input_image.to_numpy().copy()  #full_image6.to_numpy().copy()

    rois_all = roi_grid_ac[:] #[[234, 431, 50, 50], [234, 431, 10, 10]]

#############3  Draw  a rectangle and show it in fig
#drawing_image = image_copy.copy()
    drawing_image = np.ascontiguousarray(image_copy, dtype=np.float64)
    color = (0, 0, 0)    # originally (255, 0, 1)

#For drawing multiple ROIs
    imageW_roi = [ cv2.rectangle(drawing_image,(rois_all[j][0], rois_all[j][1]),\
                                 (rois_all[j][0]+ rois_all[j][2], rois_all[j][1]+ rois_all[j][3]),\
                                     color,1) for j in range(len(rois_all)) ]

    alpha =contr_alpha #1 #0.3 #1 #10#0.3 # Contrast control (1.0-3.0)
    beta = contr_beta#20#20 #5 Brightness control (0-100)

    adjusted = [cv2.convertScaleAbs(imageW_roi[k], alpha=alpha, beta=beta) for k in range(len(imageW_roi)) ]  
#adjusted_single = cv2.convertScaleAbs(imageW_roi_single, alpha=alpha, beta=beta) 
                     
    plt.figure()
    [plt.imshow(adjusted[i], cmap = 'viridis') for i in range(len(adjusted))  ]
    #plt.colorbar()
    #plt.title('{}{}{}{}{}{}'.format('[(x0,y0),(xn,yn)]', xt[0], yt[0], xt[-1], yt[-1], gridsize))
    plt.tight_layout()
    
    return roi_grid_ac




def timestamp_for_rois(inputdf, input_image,  roi_selection, flattened):
    """
    Select timestamps for the choice of ROIs. 
    Returns a list of timestamps corresponding to the chosen ROIs. 
    You can choose if you want to use dataframe with flattened timestamps or dataframe with
    non-flattened (as is) timestamps.
    """
#######################
#df1 =  full_image[['x', 'y', 't']].copy(deep = True)
#df1 =  full_image[['x', 'y', 'flat_ts']].copy(deep = True)
    if flattened is True:
        df1 =  inputdf[['x', 'y', 'flat_ts']].copy(deep = True)
    elif flattened is False:
        df1 =  inputdf[['x', 'y', 'ts', 'PhNo']].copy(deep = True)

    rois_all = roi_selection[:]
    #Image1 = np.zeros((512,512), dtype=np.int32)

    x_all = []
    y_all = []
    t_all = []
    
    image_copy = input_image.to_numpy().copy()  #full_image6.to_numpy().copy()
    drawing_image = np.ascontiguousarray(image_copy, dtype=np.float64)    
    for m in range(len(rois_all)):
        temp_df =  df1[ (df1['x'] >= rois_all[m][1]) & (df1['x'] < rois_all[m][1]+rois_all[m][3] ) 
                   & (df1['y'] >= rois_all[m][0] ) & (df1['y'] < rois_all[m][0]+rois_all[m][2] )]
        x_all.append(temp_df['x'].tolist())
        y_all.append(temp_df['y'].tolist())
        if flattened is True:
            #t_all.append(temp_df['flat_ts'].tolist())
            t_all.append( list(temp_df['flat_ts'].values[~np.isnan(temp_df['flat_ts'].values)]))
        elif flattened is False:    
            #t_all.append#(temp_df['t'].tolist())
            #t_all.append( list(temp_df['ts'].values[~np.isnan(temp_df['ts'].values)]))
            t_all.append( temp_df['ts'].explode().dropna().values.astype('int64'))
      
        #roi_image = temp_df[['x','y','PhNo']].pivot(index = 'x', columns = 'y', values = 'PhNo').fillna(0)
        imagex = temp_df['x'].values
        imagey = temp_df['y'].values
        Image1 = np.zeros((512,512), dtype=np.int32)
        for k in range(len(imagex)):
            Image1[imagex[k], imagey[k]] = Image1[imagex[k], imagey[k]] +1
        
        plt.figure()
        plt.imshow(Image1) 
        plt.imshow(drawing_image, cmap = 'viridis', vmin = 0, vmax = 300, 
                       alpha = 0.3)
       
    
    # for k in range(len(x_all[0])):
    #     Image1[ int(x_all[0][k]), int(y_all[0][k])] = Image1[ int(x_all[0][k]), int(y_all[0][k])] +1

    # plt.figure()
    # plt.imshow(Image1, cmap = 'viridis', vmin = 0, vmax = 300)
    # plt.title('First ROI in grid ')
    # print('{}{}{}{}'.format('len(t_all) = ', len(t_all), ' # of ROIs ', len(rois_all)  ))
    return t_all



def movie_maker(df_in, timewidth, timedivide, name_des):
    description = name_des
    df2 = df_in.sort_values(by = ['t'], ignore_index = True)
    df2['t'] = df2['t'].fillna(0)*200
#import numpy as np

#### I am keeping everything in nanosecond format. if time_divider = 1e9 then it becomes
## time in seconds. But I am keeping it as it is in nanosecond integer format as 
## comes out from detecor. Only multiplied by 200 ns. 
#Chose discrete_time_width to get your bin size. For instance, 1e9 would be 1s bin, 1e8 is 0.1s bin
# 1e7 would be 0.01 second bin and so on. You don't need to change time_divider, keep it 1, change
# discrete_time_width instead. 
    discrete_time_width1 =  timewidth #1e9 #1e8 * 50e3
    time_divider = timedivide #1
    time_width1 = (df2['t'].max() - df2['t'].min())/time_divider # origianlly 1e8
    print(time_width1)

    x1 = (df2['1did'].values)%512
    y1 = (df2['1did'].values)//512
    timestamp1 = df2['t'].values
    timestamp2 = timestamp1//time_divider  # origianlly 1e8
    discrete_times1 = (timestamp2 // discrete_time_width1).astype(int)

    image_shape1 = (300, 300)
    shape1 = (timestamp1.max(), *image_shape1)

    sparse_data1 = sparse.COO(( discrete_times1- discrete_times1.min(), x1, y1 ), 
                          np.ones((len(timestamp2))))

    print(sparse_data1)

    #dense_data = sparse_data1.todense()
#print(len(xx1))
    #plt.figure()
    #plt.imshow(dense_data[5], vmin = 0, vmax = 10)
    #plt.colorbar()
    return sparse_data1
#     mean_inten = [np.mean(dense_data[i]) for i in range(len(dense_data))]
#     plt.figure()
#     plt.plot(mean_inten)

# # from astropy.utils.data import get_pkg_data_filename
# # from astropy.io import fits
#     hdu = fits.PrimaryHDU(dense_data)

#     #output_file_name = expression.findall(thefilepath)[0]+'_'+bin_len+'.'+'fits'
#     output_file_name = "{}{}{}".format(description ,".", "fits")
#     hdul = fits.HDUList([hdu])
#     hdul.writeto(output_file_name)

def sparse_to_movie(sparse_data_in, filename, mean_curve):
    dense_data = sparse_data_in.todense()

    plt.figure()
    plt.imshow(dense_data[5], vmin = 0, vmax = 10)
    plt.colorbar()
 
    hdu = fits.PrimaryHDU(dense_data)
    description = filename
    #output_file_name = expression.findall(thefilepath)[0]+'_'+bin_len+'.'+'fits'
    output_file_name = "{}{}{}".format(description ,".", "fits")
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_file_name)
    
    if mean_curve is True:
        mean_inten = [np.mean(dense_data[i]) for i in range(len(dense_data))]
        plt.figure()
        plt.plot(mean_inten)
    else:
        pass



def PhotonTimeStampBinned(x, divisionNo):
    pix1ns = np.array(x, dtype = 'int64') #* 200
    pix1sort = np.sort(pix1ns)
    divide_by = divisionNo
    num_bins = int((pix1sort[-1] - pix1sort[0])/divide_by)    #1e9 gives 10 s bin width # 1e8 gives 999 bins 1s width
    each_bin_duration = int((pix1sort[-1] - pix1sort[0])/num_bins)/1e9
    print('No. of bins: ', num_bins, 'EachBinDuration: ', each_bin_duration)
    bin_edges = np.linspace(pix1sort[0], pix1sort[-1], num_bins)
    #To remove the counts from the first and last beans, they could
    # somethimes have zero values.
    binned_data= np.digitize(pix1sort, bin_edges[1:-1]) 
    # Calculate histogram counts using bin count
    hist_binned_data = np.bincount(binned_data)
    return hist_binned_data, bin_edges









# ###########################
# def raw_imager(inputdf, contrast0, contrast1):

# ## You can either input flattened_df or raw dataframe for imaging. Just input the desired one    
# ####################################################
# ### 2D (X,Y) coordinates are saved as 1D coordinates, ind = Y*512 + X. Getting them back to x,y coord
# #X2 = X.groupby('1did').apply(lambda x: x['index1'].values.min()).reset_index()

# ## To create the plot, all the photons arriving at a particular index or (x,y) coordinates need
# ## to be calculated. Group by the index 'idid' and gather all photon timestamps in that index.
# ## then find out how many timestamps there are at each index (or pixel). 
#     #X4 = flattened_df.groupby('1did').apply(lambda x: x['t'].values.tolist()).reset_index(name = 'ts')
#     X4 = inputdf.groupby('1did').apply(lambda x: x['t'].values.tolist()).reset_index(name = 'ts')
#     X4['PhNo'] = X4['ts'].apply(lambda x: len(x))
#     X4['x'] = X4['1did']%512
#     X4['y'] = X4['1did']//512

#     X5 = X4[['x', 'y', 'PhNo']]

# ### To plot have to use pivot
#     X6 = X5.pivot(index='x', columns='y', values='PhNo').fillna(0)

#     plt.figure()
#     plt.imshow(X6, vmin = contrast0, vmax = contrast1)
#     plt.colorbar()

# def mask_remover(inputdf, contrast0, contrast1):
#     """
#     The data from detector can have masks. This function converts it to 512x512 for ease of 
#     calculating q values, roi, and so on. It returns a data frame with x,y,t information
#     and also returns an image dataframe cossresponding to that. 
#     """   
# ############ Create a canvas of sorts where all 512x512 pixels would have zeros
#     full_1d_ids = []

#     for x in range(512):
#         for y in range(512):
#             tempid = y*512 + x
#             full_1d_ids.append(tempid)

# ### create zeros for all 512x512 pixels
#     zero_photon_nos = np.zeros(len(full_1d_ids))
# #Df with 100 pixel ids and phohton no which is zero
#     blank_canvas = pd.DataFrame({'1did': full_1d_ids, 'PhNo': zero_photon_nos})

#     blank_canvas2 = blank_canvas.copy(deep = True)
#     blank_canvas2['x'] = blank_canvas2['1did']%512
#     blank_canvas2['y'] = blank_canvas2['1did']//512
#     blank_canvas6 = blank_canvas2[['x','y', 'PhNo']].pivot(index = 'x', columns = 'y', values = 'PhNo')

#     ### It should be all zeros
#     plt.figure()
#     plt.imshow(blank_canvas6)
#     plt.title('Blank Canvas')

# #############################################
# ### Find the elements in canvasdf (the 512x512 big matrix) that are not inside the actual recorded data (
# # (the pixels that have photon counts )

#     #pixels_with_zeroPh = list(set(blank_canvas2['1did'].values) - set(X['1did'].values ))
#     #pixels_with_zeroPh = list(set(blank_canvas2['1did'].values) - set(flattened_df['1did'].values ))
#     pixels_with_zeroPh = list(set(blank_canvas2['1did'].values) - set(inputdf['1did'].values ))
# #These pixels are masked so they shoiuld not have any photons
#     fill_with_zero = np.zeros(len(pixels_with_zeroPh))

# ### Create a blank dataframe that has no photon counts at any  pixel
#     blank_df_allZero = pd.DataFrame({'1did': list(pixels_with_zeroPh), 'PhNo': fill_with_zero })

# ### Now combine the trdf and canvas (pixels with photon counts and without) to create full image
# #Previously with non-flattened timestamps
# #full_image = pd.concat([blank_df_allZero, X], ignore_index= True)
# ### With flattened time stamps
#     #full_image = pd.concat([blank_df_allZero, flattened_df], ignore_index= True)
#     full_image = pd.concat([blank_df_allZero, inputdf], ignore_index= True)

#     full_image['x'] = full_image['1did']%512
#     full_image['y'] = full_image['1did']//512
#     full_image4 = full_image.groupby('1did').apply(lambda x: x['t'].values.tolist()).reset_index(name = 'ts')
#     full_image4['PhNo'] = full_image4['ts'].apply(lambda x: len(x))
#     full_image4['x'] = full_image4['1did']%512
#     full_image4['y'] = full_image4['1did']//512

#     full_image5 = full_image4[['x', 'y', 'PhNo']]
#     full_image6 = full_image5.pivot(index = 'x', columns = 'y', values = 'PhNo').fillna(0)

#     plt.figure()
#     plt.imshow(full_image6, vmin = contrast0,  vmax = contrast1)
#     plt.title('full_image6_with_mask_removed')
    
#     return full_image, full_image6


# def grid_creator(topleft_x, dx, topleft_y, dy, gridsize, input_image, contr_alpha, contr_beta):

#     """
#     Creates a MxN grid. Both dx and dy should be divisible by gridsize.
#     x and y are what you read from matplotlib image directly.
#     top_left_list and bottom_right_list should have the same length.
#     input_image to draw the ROIs upon. 
#     returns the coordinates to create rois in the grid.
#     """   
# #     ########### Create a Grid of ROIs
# # xt = np.arange(440, 440+75, 1) #(125, 125+120, 1)
# # yt = np.arange(445, 445+30, 1) #(100, 100+120, 1)
#     xt = np.arange(topleft_x, topleft_x + dx, 1) #(125, 125+120, 1)
#     yt = np.arange(topleft_y, topleft_y + dy, 1) #(100, 100+120, 1)

#     gs = gridsize # grid size

#     top_left_list = [(x,y) for x in range( min(xt),
#                                       max(xt)+1, gs) for y in range(min(yt),max(yt)+1, gs) ]
#     bottom_right_list = [(x,y) for x in range( min(xt)+ (gs-1) , 
#                                           max(xt)+1, gs) for y in range(min(yt)+ (gs-1) ,
#                                                                         max(yt)+1,gs) ]
#     print("{}{}{}{}".format('top_left_list: ', len(top_left_list), 
#                             ' bottom_right_list: ', len(bottom_right_list)) )

#     rxy_grid = [[top_left_list[i][0], top_left_list[i][1],
#              bottom_right_list[i][0], bottom_right_list[i][1]] for i in range(len(top_left_list))]
# ### originally
# #rxy_grid_ac = [ [el[1], el[0],  abs(el[3] - el[1]), abs(el[2]-el[0]) ] for el in rxy_grid]
#     rxy_grid_ac = [ [el[0], el[1],  abs(el[2] - el[0]), abs(el[3]-el[1]) ] for el in rxy_grid]

# # coordinates of the upper-left corner and width and height of each rectangle
#     roi_grid_ac = np.array((rxy_grid_ac),
#                         dtype=np.int64)

#     shape = input_image.shape  #(512,512)  #full_image.shape #(512,512)  # data.shape[1:]
# # to the order they are specified in coords.
#     label_array_grid_ac = roi.rectangles(roi_grid_ac, shape)
#     roi_grid_inds_ac, pixel_grid_list_ac = roi.extract_label_indices(
#     label_array_grid_ac)

# ########################## NOw the plotting of the ROIs

#     image_copy = input_image.to_numpy().copy()  #full_image6.to_numpy().copy()

#     rois_all = roi_grid_ac[:] #[[234, 431, 50, 50], [234, 431, 10, 10]]

# #############3  Draw  a rectangle and show it in fig
# #drawing_image = image_copy.copy()
#     drawing_image = np.ascontiguousarray(image_copy, dtype=np.float64)
#     color = (0, 0, 0)    # originally (255, 0, 1)

# #For drawing multiple ROIs
#     imageW_roi = [ cv2.rectangle(drawing_image,(rois_all[j][0], rois_all[j][1]),\
#                                  (rois_all[j][0]+ rois_all[j][2], rois_all[j][1]+ rois_all[j][3]),\
#                                      color,1) for j in range(len(rois_all)) ]

#     alpha =contr_alpha #1 #0.3 #1 #10#0.3 # Contrast control (1.0-3.0)
#     beta = contr_beta#20#20 #5 Brightness control (0-100)

#     adjusted = [cv2.convertScaleAbs(imageW_roi[k], alpha=alpha, beta=beta) for k in range(len(imageW_roi)) ]  
# #adjusted_single = cv2.convertScaleAbs(imageW_roi_single, alpha=alpha, beta=beta) 
                     
#     plt.figure()
#     [plt.imshow(adjusted[i], cmap = 'viridis') for i in range(len(adjusted))  ]
#     #plt.colorbar()
#     plt.title('{}{}{}{}{}{}'.format('[(x0,y0),(xn,yn)]', xt[0], yt[0], xt[-1], yt[-1], gridsize))
#     plt.tight_layout()
    
#     return roi_grid_ac


# def roi_creator(roi_list,  input_image, contr_alpha, contr_beta):

#     """
#     Creates a list of square or rectangular rois.
#     input list like this [[x1,y1, dx1, dy1], [x2,y2,dx2,dy2],.....]
#     x and y as in the matplotlib plot
#     """   
    
# # coordinates of the upper-left corner and width and height of each rectangle
#     roi_grid_ac = np.array((roi_list),
#                         dtype=np.int64)

#     shape = input_image.shape  #(512,512)  #full_image.shape #(512,512)  # data.shape[1:]
# # to the order they are specified in coords.
#     label_array_grid_ac = roi.rectangles(roi_grid_ac, shape)
#     roi_grid_inds_ac, pixel_grid_list_ac = roi.extract_label_indices(
#     label_array_grid_ac)

# ########################## NOw the plotting of the ROIs

#     image_copy = input_image.to_numpy().copy()  #full_image6.to_numpy().copy()

#     rois_all = roi_grid_ac[:] #[[234, 431, 50, 50], [234, 431, 10, 10]]

# #############3  Draw  a rectangle and show it in fig
# #drawing_image = image_copy.copy()
#     drawing_image = np.ascontiguousarray(image_copy, dtype=np.float64)
#     color = (0, 0, 0)    # originally (255, 0, 1)

# #For drawing multiple ROIs
#     imageW_roi = [ cv2.rectangle(drawing_image,(rois_all[j][0], rois_all[j][1]),\
#                                  (rois_all[j][0]+ rois_all[j][2], rois_all[j][1]+ rois_all[j][3]),\
#                                      color,1) for j in range(len(rois_all)) ]

#     alpha =contr_alpha #1 #0.3 #1 #10#0.3 # Contrast control (1.0-3.0)
#     beta = contr_beta#20#20 #5 Brightness control (0-100)

#     adjusted = [cv2.convertScaleAbs(imageW_roi[k], alpha=alpha, beta=beta) for k in range(len(imageW_roi)) ]  
# #adjusted_single = cv2.convertScaleAbs(imageW_roi_single, alpha=alpha, beta=beta) 
                     
#     plt.figure()
#     [plt.imshow(adjusted[i], cmap = 'viridis') for i in range(len(adjusted))  ]
#     #plt.colorbar()
#     #plt.title('{}{}{}{}{}{}'.format('[(x0,y0),(xn,yn)]', xt[0], yt[0], xt[-1], yt[-1], gridsize))
#     plt.tight_layout()
    
#     return roi_grid_ac

# def timestamp_for_rois(inputdf, roi_selection, flattened):
#     """
#     Select timestamps for the choice of ROIs. 
#     Returns a list of timestamps corresponding to the chosen ROIs. 
#     You can choose if you want to use dataframe with flattened timestamps or dataframe with
#     non-flattened (as is) timestamps.
#     """
# #######################
# #df1 =  full_image[['x', 'y', 't']].copy(deep = True)
# #df1 =  full_image[['x', 'y', 'flat_ts']].copy(deep = True)
#     if flattened is True:
#         df1 =  inputdf[['x', 'y', 'flat_ts']].copy(deep = True)
#     elif flattened is False:
#         df1 =  inputdf[['x', 'y', 't']].copy(deep = True)

#     rois_all = roi_selection[:]
#     Image1 = np.zeros((512,512), dtype=np.int32)

#     x_all = []
#     y_all = []
#     t_all = []

#     for m in range(len(rois_all)):
#         temp_df =  df1[ (df1['x'] >= rois_all[m][1]) & (df1['x'] < rois_all[m][1]+rois_all[m][3] ) 
#                    & (df1['y'] >= rois_all[m][0] ) & (df1['y'] < rois_all[m][0]+rois_all[m][2] )]
#         x_all.append(temp_df['x'].tolist())
#         y_all.append(temp_df['y'].tolist())
#         if flattened is True:
#             #t_all.append(temp_df['flat_ts'].tolist())
#             t_all.append( list(temp_df['flat_ts'].values[~np.isnan(temp_df['flat_ts'].values)]))
#         elif flattened is False:    
#             t_all.append#(temp_df['t'].tolist())
#             t_all.append( list(temp_df['t'].values[~np.isnan(temp_df['t'].values)]))

        
#     for k in range(len(x_all[0])):
#         Image1[ int(x_all[0][k]), int(y_all[0][k])] = Image1[ int(x_all[0][k]), int(y_all[0][k])] +1

#     plt.figure()
#     plt.imshow(Image1, cmap = 'viridis', vmin = 0, vmax = 300)
#     plt.title('First ROI in grid ')
#     print('{}{}{}{}'.format('len(t_all) = ', len(t_all), ' # of ROIs ', len(rois_all)  ))
#     return t_all


# def ac_calc(timestamps, n_casc, n_c, divby, rois_selection): 

#     t_all = timestamps
#     N_casc = n_casc #32 #32 # 10 number of cascades
#     N_c =  n_c  # 8   # 16 number of correlation times (delay times) per cascade

#     t_tac = 5120 # this is the t_TAC in the paper

#     k_range = N_casc * N_c   # The total number of delay times

#     tau_list = []  
#     floor_thing = []
# ########### Tau according to  the rsi paper....My original Taus
# ####################################################
#     for k in range(0,k_range):
#         if k == 0:
#             tc = 1
#             tau_list.append(tc)
#         elif k > 0 and k < 2*N_c:
#             tc = tau_list[k-1] +1
#             tau_list.append(tc)
#         elif k >= (2 * N_c):
#             tc = tau_list[k-1] + 2**np.floor( (k/N_c) -1)
#             tau_list.append(tc)
#             floor_thing.append(2**np.floor( (k/N_c) -1))

#     tau_array = np.array(tau_list)   # Delay times to be used

#     tau = tau_array[:] * 1  # Delay times in integer multiples of 1280
#     #import pycorrelate as pyc
#     #shtr_cal = 179999
#     #tau_array = np.array(tau_list)   # Delay times to be used
#     #tau = tau_array[:]*1  ########

# ####################
# #tmp_y =  np.round( np.sort(t_all_arr)/200, 0)

# ####### Previous way
# #tmp_u = [np.unique(t_all[k]) for k in range(len(t_all))]
# #tmp_y =  [np.round(tmp_u[n]/5120, 0) for n in range(len(t_all))]
# ######## New Way
#     t_all_arr = [np.array(t_all[i])*divby for i in range(len(t_all))]
# #tmp_y =  [np.round( np.sort(t_all_arr[n])/200, 0) for n in range(len(t_all_arr))]
#     tmp_y =  [np.sort( np.round(t_all_arr[n])/divby, 0) for n in range(len(t_all_arr))]
# #tmp_y =  [np.sort(t_all_arr[n]) for n in range(len(t_all_arr))]

# #duration = [ max(t_all[m]) - min(t_all[m])  for m in range(len(t_all)) ]
#     duration = [ max(tmp_y[m]) - min(tmp_y[m])  for m in range(len(tmp_y)) ]                                                                                                                          
#     idtau = [np.where( duration[i] - tau > 0 )  for i in range(len(duration)) ]
#     tau_mod = [tau[idtau[j]] for j in range(len(idtau))]

#     G = [pyc.pcorrelate(tmp_y[a], tmp_y[a], tau, normalize = False) for a in range(len(tmp_y))]

#     plt.figure()
#     plt.plot( (tau[2:]*divby)/1e9, G[0][1:], 'o-')
#     plt.xscale('log')
#     plt.tight_layout()

#     def gn_calc(delay, time_stamps):
#         GN_list = []
#         for b in range(len(time_stamps)):
#             try:
#                 tmp_G = pyc.pcorrelate(time_stamps[b], time_stamps[b], delay[b], normalize = True)
#                 GN_list.append(tmp_G)
#             except ZeroDivisionError:
#                 print('{}{}'.format('ZeroDivisionEncountered at:=', b))
#                 pass
#                 tmp_G = np.zeros(len(delay[b]))
#                 GN_list.append(tmp_G[1:])
#         return GN_list       

#     GN_org = gn_calc(tau_mod, tmp_y)
# #GN_org = [pyc.pcorrelate(tmp_y[b], tmp_y[b], tau_mod[b], normalize = True) for b in range(len(tmp_y))]

#     plt.figure()
#     plt.plot( (tau_mod[0][2:]*divby)/1e9, GN_org[0][1:], '-', label ="{}{}".format(rois_selection[0],'roi1') )
#     plt.plot( (tau_mod[1][2:]*divby)/1e9, GN_org[1][1:], '-', label ="{}{}".format(rois_selection[1],'roi2'))
#     plt.xscale('log')
#     plt.legend()
#     plt.tight_layout()

#     return tau, G, tau_mod, GN_org


# def ac_plotter_grid(delay, acs, divby, logscale):

#     row = 5
#     col = len(acs)//row
#     fig, ax = plt.subplots(row, col, figsize = (24,12), sharey=True)
#     k = 0
# #while k<len(result_input[1]):
#     while k<(row*col):    
#         for i in range(0,row):
#             for j in range(0,col):
#                 if logscale is True:
#                     ax[i,j].semilogx( (delay[k][2:]*divby)/1e9, acs[k][1:], '-', 
#                          label ="{}".format(k))
#                 if logscale is False:
#                     ax[i,j].plot( (delay[k][2:]*divby)/1e9, acs[k][1:], '-', 
#                          label ="{}".format(k))
                
#                 ax[i,j].legend(frameon = False)
                
#                 #ax[i,j].set_xlim([3.6e-7, 1])
#                 ax[i,j].set_ylim([-0.9, 6])
#                 ax[i,j].grid()
#                 ax[i,j].axhline(y = 1, color = 'r', linestyle = '--', alpha = 0.4)
                
#                 k+=1
    
#     plt.tight_layout()


# def q_roi_viewer(imagefile, center_beam, distance, rad_ini, rad_fin, agrid_ini, agrid_fin,
#                  vm1, vm2):

#     center = center_beam #(201, 213)   #(300,223)  
#     binning = 1
#     pixel_size = (binning*55.0e-6, binning*55.0e-6)
#     sample_distance = 1 #1.0 #m
#     wavelength = (1240/706) #1e-9
#     shape = (512,512) # imagefile.shape #(512,512)  #Img[0:512,0:512].shape #data[0].shape

#     rad_grid = utils.radial_grid(center=center, shape=shape, pixel_size=pixel_size)
#     twotheta_grid = utils.radius_to_twotheta(dist_sample=sample_distance, radius=rad_grid)
#     q_grid = utils.twotheta_to_q(two_theta=twotheta_grid, wavelength=wavelength)
#     angle_grid = utils.angle_grid(center=center, shape=shape, pixel_size=pixel_size)

#     #df1 = masktest[0].copy(deep = True)

#     df_Img = imagefile #masktest[1] #pd.DataFrame(Img[0:512, 0:512]) # Just the image. Here row and column
# # indices are nothing but the pixel cooridiantes, i.e., x = (0 -511) and y = (0-511)
#     df_q = pd.DataFrame(q_grid)  # converts the xy pixel coordinates into qx qy
#     df_rad = pd.DataFrame(rad_grid) # Radial distance from the center
#     df_2theta = pd.DataFrame(twotheta_grid) # Two theta
#     df_angle = pd.DataFrame(angle_grid)  # Angle 

#     tgf = [df_Img, df_q, df_rad, df_2theta, df_angle]
#     full = pd.concat(tgf, axis = 1, keys = ['Pixel', 'qgrid', 'rad', '2theta', 'agrid'],  ignore_index= False)

#     q_ring_df = full['Pixel'][ (full['rad'] > rad_ini) & (full['rad'] < rad_fin)
#                        & (full['agrid'] > agrid_ini) & (full['agrid'] < agrid_fin)]

#     plt.figure()
#     plt.imshow(q_ring_df, vmin = 0, vmax = vm1, alpha = 1)
#     plt.imshow(df_Img, vmin = 0, vmax = vm2,  cmap = 'viridis', alpha = 0.7)
#     plt.title("{}{}{}{}{}{}{}".format(rad_ini,'_',rad_fin, '_', agrid_ini,'_', agrid_fin))



# def ac_ring_calc_mod(datafile, imagefile,  center_beam, distance, rad_ini, rad_fin, agrid_ini, agrid_fin, 
#                  tau_multi, 
#                  tau_linear, sort_ts, vmaxContrast, flattened):
#     """
#     datafile = is the actual dataframe that has all the x,y, and timestamp
#     imagefile = just the image to show ROI, qring etc.
#     Selects a q ring, collects all the timestamps within that ring, and calcualtes ACF. 
#     """
#     center = center_beam #(201, 213)   #(300,223)  
#     binning = 1
#     pixel_size = (binning*55.0e-6, binning*55.0e-6)
#     sample_distance = distance #1 #1.0 #m
#     wavelength = (1240/706) #1e-9
#     shape = (512,512) # imagefile.shape #(512,512)  #Img[0:512,0:512].shape #data[0].shape

#     rad_grid = utils.radial_grid(center=center, shape=shape, pixel_size=pixel_size)
#     twotheta_grid = utils.radius_to_twotheta(dist_sample=sample_distance, radius=rad_grid)
#     q_grid = utils.twotheta_to_q(two_theta=twotheta_grid, wavelength=wavelength)
#     angle_grid = utils.angle_grid(center=center, shape=shape, pixel_size=pixel_size)

#     #df1 = masktest[0].copy(deep = True)

#     df_Img = imagefile #masktest[1] #pd.DataFrame(Img[0:512, 0:512]) # Just the image. Here row and column
# # indices are nothing but the pixel cooridiantes, i.e., x = (0 -511) and y = (0-511)
#     df_q = pd.DataFrame(q_grid)  # converts the xy pixel coordinates into qx qy
#     df_rad = pd.DataFrame(rad_grid) # Radial distance from the center
#     df_2theta = pd.DataFrame(twotheta_grid) # Two theta
#     df_angle = pd.DataFrame(angle_grid)  # Angle 

#     tgf = [df_Img, df_q, df_rad, df_2theta, df_angle]
#     full = pd.concat(tgf, axis = 1, keys = ['Pixel', 'qgrid', 'rad', '2theta', 'agrid'],  ignore_index= False)

#     q_ring_df = full['Pixel'][ (full['rad'] > rad_ini) & (full['rad'] < rad_fin)
#                        & (full['agrid'] > agrid_ini) & (full['agrid'] < agrid_fin)]    
#     plt.figure()
#     plt.imshow(q_ring_df, vmin = 0, vmax = 500, alpha = 1)
#     plt.imshow(df_Img, vmin = 0, vmax = 800,  cmap = 'viridis', alpha = 0.7)
#     plt.title("{}{}{}{}{}{}{}".format(rad_ini,'_',rad_fin, '_', agrid_ini,'_', agrid_fin))

    
#     df_rad_1d = df_rad.stack().reset_index()
#     df_rad_1d.columns = ['xrad', 'yrad', 'radval']
#     df_rad_1d['id_rad_1d'] = df_rad_1d['yrad']*512 + df_rad_1d['xrad']

#     df_angle_1d = df_angle.stack().reset_index()
#     df_angle_1d.columns = ['xangle', 'yangle', 'angleval']
#     df_angle_1d['id_angle_1d'] = df_angle_1d['yangle']*512 + df_angle_1d['xangle']

#     df1 = datafile.copy(deep = True)
#     merged_rad = df1.merge(df_rad_1d, left_on = '1did', right_on = 'id_rad_1d')

#     merged_rad_angle = merged_rad.merge(df_angle_1d, left_on = 'id_rad_1d', right_on = 'id_angle_1d')
    
#     if flattened is True:
#         clean_merged_full = merged_rad_angle[['x', 'y', 'flat_ts', 't', '1did', 'radval', 'angleval']]
#         clean_merged = clean_merged_full[ (clean_merged_full['radval'] > rad_ini) & (clean_merged_full['radval'] < rad_fin) & (clean_merged_full['angleval'] > agrid_ini) & (clean_merged_full['angleval'] < agrid_fin)]
#         for_image = clean_merged.groupby('1did').apply(lambda x: x['flat_ts'].values.tolist() ).reset_index(name = 'ts')
#     elif flattened is False:
#         clean_merged_full = merged_rad_angle[['x', 'y', 't', '1did', 'radval', 'angleval']]
#         clean_merged = clean_merged_full[ (clean_merged_full['radval'] > rad_ini) & (clean_merged_full['radval'] < rad_fin) & (clean_merged_full['angleval'] > agrid_ini) & (clean_merged_full['angleval'] < agrid_fin)]
#         for_image = clean_merged.groupby('1did').apply(lambda x: x['t'].values.tolist() ).reset_index(name = 'ts')
            
#     for_image['PhNo'] = for_image['ts'].apply(lambda x: len(x))

#     for_image['x'] = for_image['1did']%512
#     for_image['y'] = for_image['1did']//512

#     for_image2 = for_image[['x', 'y', 'PhNo']]

#     for_image_plot = for_image2.pivot(index='x', columns='y', values='PhNo').fillna(0)
#     imagex = for_image2['x'].values
#     imagey = for_image2['y'].values
#     imageno = for_image2['PhNo'].values
#     Image1 = np.zeros((512,512), dtype=np.int32)

#     for k in range(len(imagex)):
#         Image1[imagex[k], imagey[k]] = Image1[imagex[k], imagey[k]] +1

#     plt.figure()
#     plt.imshow(Image1)
#     plt.imshow(df_Img, vmin = 0, vmax = vmaxContrast,  cmap = 'viridis', alpha = 0.7)
#     plt.title("{}{}{}{}{}{}{}".format(rad_ini,'_',rad_fin, '_', agrid_ini,'_', agrid_fin))


#     plt.figure()
#     plt.imshow(for_image_plot, vmin = 0, vmax = vmaxContrast, alpha = 1, extent =[0,512,0,512], aspect = 'auto')
#     plt.title('DirectlyFrom_for_image2')
# #plt.imshow(df_Img, alpha = 0.5)
# #plt.colorbar()

#     if tau_multi is True:
#         # # ######## Tau multitau
#         N_casc = 32 #32 # 10 number of cascades
#         N_c =  8   # 16 number of correlation times (delay times) per cascade
#         t_tac = 5120 # this is the t_TAC in the paper
#         k_range = N_casc * N_c   # The total number of delay times
#         tau_list = []  
#         floor_thing = []
#         ########### Tau according to  the rsi paper....My original Taus
#         ####################################################
#         for k in range(0,k_range):
#             if k == 0:
#                 tc = 1
#                 tau_list.append(tc)
#             elif k > 0 and k < 2*N_c:
#                 tc = tau_list[k-1] +1
#                 tau_list.append(tc)
#             elif k >= (2 * N_c):
#                 tc = tau_list[k-1] + 2**np.floor( (k/N_c) -1)
#                 tau_list.append(tc)
#                 floor_thing.append(2**np.floor( (k/N_c) -1))
        
        
#         tau_array = np.array(tau_list)   # Delay times to be used
        
#         tau = tau_array[:]*1  # Delay times in integer multiples of 1280

#     elif tau_linear is True:
#         tau_array = np.arange(1,19000,1)
#         tau = tau_array[:]*1  #######

#     if flattened is True:
#         t_all_arr = (clean_merged['flat_ts'].values)*200
#         t_all_noNan = t_all_arr[~np.isnan(t_all_arr)]
#         output_t = t_all_noNan/200
#     if flattened is False:
#         t_all_arr = (clean_merged['t'].values)*200
#         t_all_noNan = t_all_arr[~np.isnan(t_all_arr)]
#         output_t = t_all_noNan/200
#     if sort_ts is True:
#         tmp_y =  np.sort( list(map(int, t_all_noNan/200 )))
#         #tmp_y =  np.sort(np.round( np.array(t_all_arr)/200, 0))
#     elif sort_ts is False:
#         tmp_y = list(map(int, t_all_noNan/200 ))
#         #tmp_y =  np.round( np.array(t_all_arr)/200, 0)
        
#     duration = max(tmp_y) - min(tmp_y)                                                                                                                         
#     idtau = ( duration - tau > 0 )
#     tau_mod = tau[idtau]       
#     roi_duration = np.round( ((max(t_all_noNan) - min(t_all_noNan)))/1e9)
#     total_events = len(t_all_noNan)
#     G = pyc.pcorrelate(tmp_y, tmp_y, tau, normalize = False)
#     #G = pyc.pcorrelate(t_all_arr, t_all_arr, tau, normalize = False)    
#     plt.figure()
#     plt.plot( (tau[2:]*200)/1e9, G[1:], '-', label = 'roi1')
#     plt.xscale('log')
#     plt.legend()
#     plt.title('{}{}{}{}'.format('ExpDurationROI_TotalEvents_',roi_duration, '_', total_events))
#     plt.tight_layout()
#     #plt.show()

#     GN_org = pyc.pcorrelate(tmp_y, tmp_y, tau_mod, normalize = True)

#     plt.figure()
#     plt.plot( (tau_mod[2:]*200)/1e9, GN_org[1:], 'o-', label ="{}{}".format('q','signal') )
#     plt.xscale('log')
#     plt.title('{}{}{}{}'.format('ExpDurationROI_TotalEvents_',roi_duration, '_', total_events))
#     plt.legend()
#     plt.tight_layout()
    
#    #clean_merged['flat_ts'].values may have NaN values better check them
#     return tau, G, tau_mod, GN_org,  rad_ini, rad_fin, agrid_ini, agrid_fin,\
#         roi_duration, total_events, output_t




# def timestamps_from_ring(datafile, imagefile,  center_beam, distance, rad_ini, rad_fin, agrid_ini, agrid_fin, 
#                  flattened):
#     """
#     datafile = is the actual dataframe that has all the x,y, and timestamp
#     imagefile = just the image to show ROI, qring etc.
#     Selects a q ring, collects all the timestamps within that ring, and calcualtes ACF. 
#     """
#     center = center_beam #(201, 213)   #(300,223)  
#     binning = 1
#     pixel_size = (binning*55.0e-6, binning*55.0e-6)
#     sample_distance = distance #1 #1.0 #m
#     wavelength = (1240/706) #1e-9
#     shape = (512,512) # imagefile.shape #(512,512)  #Img[0:512,0:512].shape #data[0].shape

#     rad_grid = utils.radial_grid(center=center, shape=shape, pixel_size=pixel_size)
#     twotheta_grid = utils.radius_to_twotheta(dist_sample=sample_distance, radius=rad_grid)
#     q_grid = utils.twotheta_to_q(two_theta=twotheta_grid, wavelength=wavelength)
#     angle_grid = utils.angle_grid(center=center, shape=shape, pixel_size=pixel_size)

#     #df1 = masktest[0].copy(deep = True)

#     df_Img = imagefile #masktest[1] #pd.DataFrame(Img[0:512, 0:512]) # Just the image. Here row and column
# # indices are nothing but the pixel cooridiantes, i.e., x = (0 -511) and y = (0-511)
#     df_q = pd.DataFrame(q_grid)  # converts the xy pixel coordinates into qx qy
#     df_rad = pd.DataFrame(rad_grid) # Radial distance from the center
#     df_2theta = pd.DataFrame(twotheta_grid) # Two theta
#     df_angle = pd.DataFrame(angle_grid)  # Angle 

#     tgf = [df_Img, df_q, df_rad, df_2theta, df_angle]
#     full = pd.concat(tgf, axis = 1, keys = ['Pixel', 'qgrid', 'rad', '2theta', 'agrid'],  ignore_index= False)

#     q_ring_df = full['Pixel'][ (full['rad'] > rad_ini) & (full['rad'] < rad_fin)
#                        & (full['agrid'] > agrid_ini) & (full['agrid'] < agrid_fin)]    
#     plt.figure()
#     plt.imshow(q_ring_df, vmin = 0, vmax = 500, alpha = 1)
#     plt.imshow(df_Img, vmin = 0, vmax = 800,  cmap = 'viridis', alpha = 0.7)
#     plt.title("{}{}{}{}{}{}{}".format(rad_ini,'_',rad_fin, '_', agrid_ini,'_', agrid_fin))

    
#     df_rad_1d = df_rad.stack().reset_index()
#     df_rad_1d.columns = ['xrad', 'yrad', 'radval']
#     df_rad_1d['id_rad_1d'] = df_rad_1d['yrad']*512 + df_rad_1d['xrad']

#     df_angle_1d = df_angle.stack().reset_index()
#     df_angle_1d.columns = ['xangle', 'yangle', 'angleval']
#     df_angle_1d['id_angle_1d'] = df_angle_1d['yangle']*512 + df_angle_1d['xangle']

#     df1 = datafile.copy(deep = True)
#     merged_rad = df1.merge(df_rad_1d, left_on = '1did', right_on = 'id_rad_1d')

#     merged_rad_angle = merged_rad.merge(df_angle_1d, left_on = 'id_rad_1d', right_on = 'id_angle_1d')
    
#     if flattened is True:
#         clean_merged_full = merged_rad_angle[['x', 'y', 'flat_ts', 't', '1did', 'radval', 'angleval']]
#         clean_merged = clean_merged_full[ (clean_merged_full['radval'] > rad_ini) & (clean_merged_full['radval'] < rad_fin) & (clean_merged_full['angleval'] > agrid_ini) & (clean_merged_full['angleval'] < agrid_fin)]
#         #for_image = clean_merged.groupby('1did').apply(lambda x: x['flat_ts'].values.tolist() ).reset_index(name = 'ts')
#     elif flattened is False:
#         clean_merged_full = merged_rad_angle[['x', 'y', 't', '1did', 'radval', 'angleval']]
#         clean_merged = clean_merged_full[ (clean_merged_full['radval'] > rad_ini) & (clean_merged_full['radval'] < rad_fin) & (clean_merged_full['angleval'] > agrid_ini) & (clean_merged_full['angleval'] < agrid_fin)]
#         #for_image = clean_merged.groupby('1did').apply(lambda x: x['t'].values.tolist() ).reset_index(name = 'ts')
            

#     if flattened is True:
#         t_all_arr = (clean_merged['flat_ts'].values)*200
#         t_all_noNan = t_all_arr[~np.isnan(t_all_arr)]
#         output_t = t_all_noNan/200
#     if flattened is False:
#         t_all_arr = (clean_merged['t'].values)*200
#         t_all_noNan = t_all_arr[~np.isnan(t_all_arr)]
#         output_t = t_all_noNan/200
    
    
#    #clean_merged['flat_ts'].values may have NaN values better check them
#     return output_t









