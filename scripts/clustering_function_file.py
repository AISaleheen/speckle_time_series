#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# From C:\Berkeley\DATA\june23_24\clustering_etc\chatgpt2


# In[1]:


# -*- coding: utf-8 -*-
#import skbeam
import sys
sys.path.append('C:\Berkeley\DATA\cosmic_tpx3\module_file')

import tpx3_module as tpx3
import tpx3_module_2 as tpx3_2

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

import struct as struct
import pandas as pd 
from astropy.io import fits
import sys
import re
import os
from pandas.plotting import autocorrelation_plot
import multipletau
from tqdm.notebook import trange, tqdm
from skimage import measure, color, io
from scipy import ndimage
import matplotlib.patches as patches
from PIL import Image
from scipy.stats import t


# In[2]:



from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS, TapTool, Slider, Div
from bokeh.plotting import figure, show #Figure
from bokeh.palettes import Magma, Inferno, Plasma, Viridis256, Cividis

from bokeh.models import Label, LabelSet, Range1d, Span



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


# In[25]:


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
    plt.figure(figsize = (9,6))
    plt.subplot(121)
#plt.imshow(normalized_array[300:500, 0:500], cmap = 'gray')
    plt.imshow(normalized_array, cmap = 'gray')
    plt.colorbar(fraction=0.046, pad=0.08)

    plt.subplot(122)
    #plt.figure()
    plt.hist(normalized_array.flatten())
    return normalized_array

#normalized_array_full = full_image_for_speckle_detection(mask_remover250[1])


# In[5]:


def qring_image_for_speckle_detection(datafile, imagefile,  center_beam, distance, 
                             rad_ini, rad_fin, agrid_ini, agrid_fin, 
                 flattened = False, after_extraction_plot = False):
    """
    Input the qring dataframe, and outputs a normalized image of the qring 
    that can be used for speckle detection within the q-ring
    """
    q250_1_ts =tpx3_2.timestamps_from_ring_movie(datafile, imagefile, center_beam, distance, 
                         rad_ini, rad_fin, agrid_ini, agrid_fin,
                         flattened = False, after_extraction_plot=False)
    return q250_1_ts


# In[6]:




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
    plt.figure(figsize = (9,6))
    plt.subplot(121)
    plt.imshow(normalized_array, cmap = 'gray')
    plt.colorbar(fraction=0.046, pad=0.08)
    plt.subplot(122)
    plt.hist(normalized_array.flatten())
    plt.figure()
    plt.imshow(qimage)
    return normalized_array, qimage, qdf

    
#q1_image_norm, q1image, q1df = qring_image_for_speckle_detection(q250_1_ts[1])


# In[7]:


# Function to identify speckles
def identify_speckles(image_in, threshold, min_size):
    # Load the image in grayscale
    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image_in
    # Apply a threshold to isolate potential speckles
    _, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Find connected components (blobs) #chain_approximation_none will return all points within the contour, try that
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Filter speckles based on size
    speckles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_size:  # Speckles are typically small
            speckles.append(contour)

    # Create an output image to visualize speckles
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, speckles, -1, (0, 0, 255), 2)

    return output_image, speckles #len(speckles)


# In[8]:



def speckle_finder_cv2(input_image, kernel, sd, thresh, min_size, max_size):
# Load the image
    """
    kernel should be (5,5) in this form; positive and odd
    sd is an integer 
    thresh is an integer no
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





# In[9]:


def ac_calculator_speckles(speckles_list, normalized_image_in, main_df, divide_by = 1e9, deltaT = 1):
    """
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
    
    plt.figure(figsize = (9,6))
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
         tmp_bin, tmp_edge = tpx3_2.PhotonTimeStampBinned(np.sort(value*200), divide_by)
         #binned_data_dict[key] = tmp_bin
         #bin_edges_dict[key] = tmp_edge
         temp_bin_df = pd.DataFrame({'bins': pd.Series(tmp_bin[1:-1]), 'edges': pd.Series(tmp_edge[1:-1]) }) # discard the fist and last bins
         binned_data_edges_dict[key] = temp_bin_df
    # plot the binned data, i.e., time series    
    plt.figure()
    for key,val in binned_data_edges_dict.items():        
        plt.plot(np.cumsum(np.diff(val['edges']))/divide_by, val['bins'].dropna()   )
    plt.show()
    
# calculate acf with multipletau for each binned data, that is for each speckle
    mpt_ac_dict = {}
    for key,value in binned_data_edges_dict.items():
        tmp_ac = multipletau.autocorrelate( value['bins'].dropna().astype('float64'), 
                               deltat = deltaT, normalize=True, compress = 'average')
        mpt_ac_dict[key] = tmp_ac
    # Few curves from mpt
    plt.figure()
    for key,val in mpt_ac_dict.items():
        plt.semilogx(val[:,0], val[:,1], 'o-')
    plt.title('Plots from MPT')
    plt.show()
    
    # Calcualte acfs with pandas autocorrelation_plot
    pandas_res_dict = {}
    
    for key,value in binned_data_edges_dict.items():
        tempax = autocorrelation_plot(value['bins'].dropna())
        plt.xscale('log')
        plt.title('Plot from Pandas')
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
        
        


# In[10]:


def ac_calculator_speckles_30min(speckles_list, normalized_image_in, main_df, divide_by = 1e9, deltaT = 1):
    """
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
    
    plt.figure(figsize = (9,6))
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
    
    tdf1 = main_df[0].copy(deep = True)     #tdf = main_df.copy(deep = True)   #mask_remover250[0].copy(deep = True)
    tdf2 = main_df[1].copy(deep = True)
    # Find the timestamps, photon counts etc. from the main dataframe for each set of speckle coordinates. Each speckle will be a dataframe that has
    # all the timestamps, coordinates etc.,. All speckle dataframes are going to be stored in a dict, e.g., spdict. For only the timestamps, we can
    # look at the tsdict. From now on, the keys in the dictionaries are going to be important because they identify the speckles and the data.
    speckles_dict1 = {}
    timestamps_dict1 = {}
    speckles_dict2 = {}
    timestamps_dict2 = {}
    for index, ids in enumerate(all_1did):
        #print(index, ids)
        tempdf1 = tdf1[tdf1['1did'].isin(ids)].dropna()
        speckles_dict1[index] = tempdf1
        tempdf2 = tdf2[tdf2['1did'].isin(ids)].dropna()
        speckles_dict2[index] = tempdf2
        tempts1 = tempdf1.explode('ts')['ts'].to_numpy().astype('int64')
        timestamps_dict1[index] = tempts1
        tempts2 = tempdf2.explode('ts')['ts'].to_numpy().astype('int64')
        timestamps_dict2[index] = tempts2
    
    #for (key1, value1), (key2, value2) in zip(dict1.items(), dict2.items()):

    # Now, iterate through each df in the dictionary, that is iterate through each speckle, and bin the timestamps. 
    binned_data_edges_dict = {}
    for (key1,value1), (key2,value2) in  zip(timestamps_dict1.items(), timestamps_dict2.items() ):
         #print(f"Key: {key}, Value: {value}")
         tmp_bin1, tmp_edge1 = tpx3_2.PhotonTimeStampBinned(np.sort(value1*200), divide_by)
         tmp_bin2, tmp_edge2 = tpx3_2.PhotonTimeStampBinned(np.sort(value2*200), divide_by)
         combined_bin = np.concatenate((tmp_bin1[1:-1], tmp_bin2[1:-1]))
         tmp_edge2_mod = tmp_edge1[1:-1][-1] + tmp_edge2[1:-1]   #stiching the last of the first bin edge to the second bin edge
         combined_edge = np.concatenate((tmp_edge1[1:-1], tmp_edge2_mod))
         temp_bin_df = pd.DataFrame({'bins': pd.Series(combined_bin), 'edges': pd.Series(combined_edge) }) # discard the fist and last bins
         #temp_bin_df = pd.DataFrame({'bins': pd.Series(tmp_bin[1:-1]), 'edges': pd.Series(tmp_edge[1:-1]) }) # discard the fist and last bins
         binned_data_edges_dict[key1] = temp_bin_df
    # plot the binned data, i.e., time series    
    plt.figure()
    for key,val in binned_data_edges_dict.items():        
        plt.plot( np.cumsum(np.diff(val.dropna()['edges']))/1e9, val.dropna()['bins'][:-1]               )
        #plt.plot(np.cumsum(np.diff(val['edges']))/divide_by, val['bins'].dropna()   )
    plt.show()
    
# calculate acf with multipletau for each binned data, that is for each speckle
    mpt_ac_dict = {}
    for key,value in binned_data_edges_dict.items():
        tmp_ac = multipletau.autocorrelate( value['bins'].dropna().astype('float64'), 
                               deltat = deltaT, normalize=True, compress = 'average')
        mpt_ac_dict[key] = tmp_ac
    # Few curves from mpt
    plt.figure()
    for key,val in mpt_ac_dict.items():
        plt.semilogx(val[:,0], val[:,1], 'o-')
    plt.title('Plots from MPT')
    plt.show()
    
    # Calcualte acfs with pandas autocorrelation_plot
    pandas_res_dict = {}
    
    for key,value in binned_data_edges_dict.items():
        tempax = autocorrelation_plot(value['bins'].dropna())
        plt.xscale('log')
        plt.title('Plot from Pandas')
        #tempax = autocorrelation_plot(value, label = "{}".format(key))
        #pandas_lags_dict[key] = tempax.lines[-1].get_xdata() 
        #pandas_acs_dict[key] = tempax.lines[-1].get_ydata() 
        temp_pandas_df = pd.DataFrame({'lag': tempax.lines[-1].get_xdata()*deltaT, 'ac': tempax.lines[-1].get_ydata()    })
        pandas_res_dict[key] = temp_pandas_df
    
    #plt.figure()
    #plt.semilogx(pandas_res_dict[0]['lag'], pandas_res_dict[0]['ac'])
    #plt.semilogx(pandas_res_dict[1]['lag'], pandas_res_dict[1]['ac'])
    #[plt.axhline(y = i, linestyle = '--', color = 'gray') for i in [0.091, -0.091]]
    #[plt.axhline(y = i, linestyle = '-', color = 'gray') for i in [0.068, -0.068]]
    #plt.show()
    
    #return binned_data_edges_dict, pandas_res_dict, mpt_ac_dict, speckles_dict, reduced_speckles
    return binned_data_edges_dict, pandas_res_dict, mpt_ac_dict, speckles_dict1, reduced_speckles

        
        


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
    


# In[12]:


def dynamic_speckle_finder_2(ac_dict_in, speckles_dict_in, normalized_image_in,  all_speckle_coords, xmax, hline = 0.068):
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
    
    #for key, val in ac_dict_in.items():
        #print(f"Key: {lagkey}, lagval: {lagval}, acval: {acval}")
        #subtracted_from_ci = val['ac'][val['lag'] < 30] - 0.068
        #subtracted_from_ci_avg = np.mean(subtracted_from_ci)
        #if subtracted_from_ci_avg > 0:
            #tempdf = pd.DataFrame({'lag': val['lag'], 'ac': val['ac']})
            #chosen_acs_lags_pandas[key] = tempdf
    
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


# In[15]:

from statsmodels.tsa.stattools import adfuller

def pval_plotter(dict_in, alpha_tolerance = 0.05):
    p_val_dict = {}
    res_dict = {}
    for key,val in dict_in.items():
        series = val['bins'].dropna().values
        adfuller_result = adfuller(series, autolag='AIC')
        p_val = adfuller_result[1]
        ADF_Statistic = adfuller_result[0]
        p_val_dict[key] = p_val
        res_dict[key] = adfuller_result
        if p_val < alpha_tolerance:
            print(f'{key}:{np.round(p_val,3)} ADF {np.round(ADF_Statistic,3)}  : Stationary')
        elif p_val >= alpha_tolerance:
            print(f'{key}:{np.round(p_val,3)} ADF {np.round(ADF_Statistic,3)} : Non-Stationary')
    return res_dict, p_val_dict
        
def find_binned_data_for_dynamic_ac(binned_data_dict, dynamic_ac_keys):
    dynamic_bin = {}
    for key in binned_data_dict.keys():
        if key in dynamic_ac_keys:
            dynamic_bin[key] = binned_data_dict[key]
    return dynamic_bin
def pval_plotter_dynamic(binned_data_dict, dynamic_ac_keys, alpha_tolerance = 0.05):
    p_val_dict = {}
    res_dict = {}
    dynamic_bin = {}
    for key in binned_data_dict.keys():
        if key in dynamic_ac_keys:
            dynamic_bin[key] = binned_data_dict[key]
    #dict_in = find_binned_data_for_dynamic_ac(all_bin_dict, dynamic_ac)
    for key,val in dynamic_bin.items():
        series = val['bins'].dropna().values
        adfuller_result = adfuller(series, autolag='AIC')
        p_val = adfuller_result[1]
        ADF_Statistic = adfuller_result[0]
        p_val_dict[key] = p_val
        res_dict[key] = adfuller_result
        if p_val < alpha_tolerance:
            print(f'{key}:{np.round(p_val,3)} ADF {np.round(ADF_Statistic,3)}  : Stationary')
        elif p_val >= alpha_tolerance:
            print(f'{key}:{np.round(p_val,3)} ADF {np.round(ADF_Statistic,3)} : Non-Stationary')
    return res_dict, p_val_dict

        

def ac_calculator_speckles_speckle_normalized(speckles_list, normalized_image_in, main_df, divide_by = 1e9, deltaT = 1, alpha_tolerance = 0.05):
    """
    collects timestamps for all speckle coordinates, create a binned data for 
    those timestamps, and use that data to normalize each speckle. 
    Also, rejects any speckle that is not stationary
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
    
    plt.figure(figsize = (9,6))
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
    
    # for normalization
    all_1did_flat = np.concatenate(all_1did).tolist()
    tempdf_normalization = tdf[tdf['1did'].isin(all_1did_flat)].dropna()
    allts_normalization =  tempdf_normalization.explode('ts')['ts'].to_numpy().astype('int64')
    tmp_bin_normalization, tmp_edge_normalization = tpx3_2.PhotonTimeStampBinned(np.sort(allts_normalization*200), divide_by)
    
    
    # Now, iterate through each df in the dictionary, that is iterate through each speckle, and bin the timestamps. 
    binned_data_edges_dict = {}
    for key, value in timestamps_dict.items():
         #print(f"Key: {key}, Value: {value}")
         tmp_bin, tmp_edge = tpx3_2.PhotonTimeStampBinned(np.sort(value*200), divide_by)
         normalized_tmp_bin = tmp_bin/tmp_bin_normalization
         adfuller_result = adfuller(normalized_tmp_bin, autolag='AIC')
         p_val = adfuller_result[1]
         if p_val < alpha_tolerance:
            temp_bin_df = pd.DataFrame({'bins': pd.Series(normalized_tmp_bin[1:-1]), 'edges': pd.Series(tmp_edge[1:-1]) }) # discard the first and last bins as there might be error in them
            binned_data_edges_dict[key] = temp_bin_df
         elif p_val >=alpha_tolerance:
            pass
            #temp_bin_df = pd.DataFrame({'bins': pd.Series(tmp_bin[1:-1]), 'edges': pd.Series(tmp_edge[1:-1]) }) # discard the first and last bins as there might be error in them
            #binned_data_edges_dict[key] = temp_bin_df
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
    plt.figure()
    for key,val in mpt_ac_dict.items():
        plt.semilogx(val[:,0], val[:,1], 'o-')
    plt.title('Plots from MPT')
    plt.show()
    
    # Calcualte acfs with pandas autocorrelation_plot
    pandas_res_dict = {}
    
    for key,value in binned_data_edges_dict.items():
        tempax = autocorrelation_plot(value['bins'].dropna() )
        plt.xscale('log')
        plt.title('Plot from Pandas')
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
        
def ac_calculator_speckles_normalizedbin(binned_data_for_norm, speckles_list, normalized_image_in, main_df, divide_by = 1e9, deltaT = 1, alpha_tolerance = 0.05):
    """
    binned_data_for_norm: input a binned data for normalization. Obviously,
    this has to be binned in the same manner as other speckles. It could be 
    data from direct beam, or a big roi, and so on. 
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
    
    plt.figure(figsize = (9,6))
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
         tmp_bin, tmp_edge = tpx3_2.PhotonTimeStampBinned(np.sort(value*200), divide_by)
         normalized_tmp_bin = tmp_bin/binned_data_for_norm
         adfuller_result_after_norm = adfuller(normalized_tmp_bin, autolag='AIC')
         p_val_after_norm = adfuller_result_after_norm[1]
         if p_val_after_norm < alpha_tolerance:
            temp_bin_df = pd.DataFrame({'bins': pd.Series(normalized_tmp_bin[1:-1]), 'edges': pd.Series(tmp_edge[1:-1]) }) # discard the first and last bins as there might be error in them
            binned_data_edges_dict[key] = temp_bin_df
         elif p_val_after_norm >=alpha_tolerance:
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
    plt.figure()
    for key,val in mpt_ac_dict.items():
        plt.semilogx(val[:,0], val[:,1], 'o-')
    plt.title('Plots from MPT')
    plt.show()
    
    # Calcualte acfs with pandas autocorrelation_plot
    pandas_res_dict = {}
    
    for key,value in binned_data_edges_dict.items():
        tempax = autocorrelation_plot(value['bins'].dropna() )
        plt.xscale('log')
        plt.title('Plot from Pandas')
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
    
    plt.figure(figsize = (9,6))
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
         tmp_bin, tmp_edge = tpx3_2.PhotonTimeStampBinned(np.sort(value*200), divide_by)
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
    plt.figure()
    for key,val in mpt_ac_dict.items():
        plt.semilogx(val[:,0], val[:,1], 'o-')
    plt.title('Plots from MPT')
    plt.show()
    
    # Calcualte acfs with pandas autocorrelation_plot
    pandas_res_dict = {}
    
    for key,value in binned_data_edges_dict.items():
        tempax = autocorrelation_plot(value['bins'].dropna() )
        plt.xscale('log')
        plt.title('Plot from Pandas')
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
        

# From com_clustered_detrended 


def dynamic_ac_arranger(keylist, dfin, dynamic_xmax):
    '''
    Finds out dynamic acs and outupts them as dictionary
    keylist: input which q values you want such as ['q1', 'q2'], and so on
    dfin: input the dictionary that has results 
    dynamic_xmax: The maximum lagtime considered to determine dynamic-ness in the ACF vs Lag data
    '''
    out_dict = {}
    cilist = []
    for key in keylist:
        # calculate the confidence interval (horizontal line) to determine if the acf is dynamic or not
        FirstBinKey = list(dfin['{}'.format(key)]['{}_bin'.format(key)].keys())[0]
        ci95 = np.round(1.96 / np.sqrt(dfin['{}'.format(key)]['{}_bin'.format(key)][FirstBinKey].dropna().shape[0]),3)

        dynamic_ac, dynamic_dict, dynamic_coords, dynamic_ci, avg_ac  = dynamic_speckle_finder_3(dfin['{}'.format(key)]['{}_pandas'.format(key)], 
                                                                 dfin['{}'.format(key)]['{}_spdict'.format(key)],  
                                                                 dfin['{}'.format(key)]['{}_image_norm'.format(key)].astype(np.uint8),  
                                                                 dfin['{}'.format(key)]['{}_allspeckles'.format(key)], dynamic_xmax, ci95 )
    
    
        out_dict['{}_dynamic_ac'.format(key)] = dynamic_ac
        out_dict['{}_dynamic_coords'.format(key)] = dynamic_coords
        out_dict['{}_avg_ac'.format(key)] = avg_ac
        out_dict['{}_dynamic_ci'.format(key)] = dynamic_ci
        cilist.append(ci95)
    return out_dict
#ft = dynamic_ac_arranger(['q1', 'q2'], res_200K, 150) 

def speckle_image_plotter(normalized_q_image_in, all_speckle_coord, dynamic_speckle_coord):
    all_speckle_image = cv2.cvtColor(normalized_q_image_in, cv2.COLOR_GRAY2BGR) #cv2.cvtColor( res_200K['q1']['q1_image_norm'].astype(np.uint8), cv2.COLOR_GRAY2BGR)
    dynamic_speckle_image = cv2.cvtColor(normalized_q_image_in, cv2.COLOR_GRAY2BGR) #cv2.cvtColor( res_200K['q1']['q1_image_norm'].astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(all_speckle_image, all_speckle_coord , -1, (255, 255, 255), 2)
    cv2.drawContours(dynamic_speckle_image, dynamic_speckle_coord , -1, (0, 255, 0), 2)
    #cv2.drawContours(all_speckle_image, res_200K['q1']['q1_allspeckles'] , -1, (0, 0, 255), 2)
    #cv2.drawContours(dynamic_speckle_image, q1_dynamic_coords200 , -1, (0, 255, 0), 2)
    return all_speckle_image, dynamic_speckle_image

#q1_200K_all, q1_200K_dynamic = speckle_image_plotter(res_200K['q1']['q1_image_norm'].astype(np.uint8),  res_200K['q1']['q1_allspeckles'], ft['q1_dynamic_coords']  )

def full_image_for_speckle_detection(image_in):
    """
    input mask_remover[1] (image array), outputs normalized array that can be used as an input to 
    the speckle_detection scheme.
    """
    img_array = image_in.to_numpy() #mask_remover[1].to_numpy()
    normalized_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min())) * 256
# Convert to integer type if needed
    normalized_array = normalized_array  #.astype(np.uint8)
    testimage = Image.fromarray(normalized_array.astype(np.uint8), 'L')
    plt.figure(figsize = (9,6))
    plt.subplot(121)
#plt.imshow(normalized_array[300:500, 0:500], cmap = 'gray')
    plt.imshow(normalized_array, cmap = 'gray')
    plt.colorbar(fraction=0.046, pad=0.08)

    plt.subplot(122)
    #plt.figure()
    plt.hist(normalized_array.flatten())
    return normalized_array
#norm200K = full_image_for_speckle_detection(images_200K['200K'])

def speckle_image_dict_maker(keylist, res_dict_in, dynamic_dict_in):
    '''
    draws speckles on normalized q-images.
    keylist: input keys ['q1', 'q2']
    res_dict_in: input the result dict to have the coordinates of all speckles
    dynamic_dict_in: have all the coords for dynamic speckles
    convert_to_int8: if need to convert the normalized q image to np.uint8
    '''
    output_image_dict = {}
    input_image_data_type = res_dict_in['{}'.format(keylist[0])]['{}_image_norm'.format(keylist[0])].dtype
    
    if input_image_data_type == 'uint8':
        for key in keylist:
            all_speckle_img, dynamic_speckle_img = speckle_image_plotter(
                res_dict_in['{}'.format(key)]['{}_image_norm'.format(key)],  
                                                     res_dict_in['{}'.format(key)]['{}_allspeckles'.format(key)], 
                                  dynamic_dict_in['{}_dynamic_coords'.format(key)]  )
            output_image_dict['{}_all_sp'.format(key)] = all_speckle_img
            output_image_dict['{}_dynamic_sp'.format(key)] = dynamic_speckle_img
    
    elif input_image_data_type != 'uint8':
         for key in keylist:
                all_speckle_img, dynamic_speckle_img = speckle_image_plotter(
                    res_dict_in['{}'.format(key)]['{}_image_norm'.format(key)].astype(np.uint8),  
                                                     res_dict_in['{}'.format(key)]['{}_allspeckles'.format(key)], 
                                  dynamic_dict_in['{}_dynamic_coords'.format(key)]  )
                output_image_dict['{}_all_sp'.format(key)] = all_speckle_img
                output_image_dict['{}_dynamic_sp'.format(key)] = dynamic_speckle_img
        
    return output_image_dict

#img_200K = speckle_image_dict_maker(['q1', 'q2', 'q3'], res_200K, dynamic_data_200K, True )

def full_image_for_speckle_detection(image_in):
    """
    input mask_remover[1] (image array), outputs normalized array that can be used as an input to 
    the speckle_detection scheme.
    """
    img_array = image_in.to_numpy() #mask_remover[1].to_numpy()
    normalized_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min())) * 256
# Convert to integer type if needed
    normalized_array = normalized_array  #.astype(np.uint8)
    testimage = Image.fromarray(normalized_array.astype(np.uint8), 'L')
    plt.figure(figsize = (9,6))
    plt.subplot(121)
#plt.imshow(normalized_array[300:500, 0:500], cmap = 'gray')
    plt.imshow(normalized_array, cmap = 'gray')
    plt.colorbar(fraction=0.046, pad=0.08)

    plt.subplot(122)
    #plt.figure()
    plt.hist(normalized_array.flatten())
    return normalized_array

#from PIL import Image
def transparent_image_creator(image_in):
    '''
    Takes the q-image with speckles in and makes the image transparent 
    everywhere except in the q-ring region. It is assumed that the values
    are zero everywhere except at the q-ring region
    '''
    array = image_in.copy()
    image = Image.fromarray(array).convert('RGBA')
    data = np.array(image)
    # Create a mask where the pixel values are zero
    mask = np.all(data[:, :, :3] == 0, axis=-1)
    # Set the alpha channel to 0 (transparent) where the mask is True
    data[mask, 3] = 0
    # The below lines are if you want to create a boundary line for the q-ring 
    mask2 = data[:, :, 3] > 0
    contours, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(data, contours, -1, (0, 255, 0, 255), 2)  # Green color with full opacity
    ############# end boundary line for the q-ring
    # Convert numpy array back to image
    transparent_image = Image.fromarray(data)
    return np.array(transparent_image)



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

def dynamic_speckle_finder_4(ac_dict_in, speckles_dict_in, normalized_image_in,  all_speckle_coords, xmax, hline = 0.068):
    """
    It can average acfs that have unequal sizes by the use of tolerant_mean function
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
        # some of the acs have different lenghts especially at shorter bins. We are using the tolerant_mean
        # function to calculate average of unequal arrays.
        average_all_acs_pandas = tolerant_mean(all_acs_pandas)[0]
        #average_all_acs_pandas = np.average(all_acs_pandas, axis = 0)
        # Now, we choose a lag wholse length is equal to the average ac.
        lag_for_plotting_avg_ac = []
        for key,val in ac_dict_in.items():
            if len(val['lag']) == len(average_all_acs_pandas):
                lag_for_plotting_avg_ac  = val['lag'].values
                break

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
        #plt.semilogx(ac_dict_in[first_key]['lag'][0:len(average_all_acs_pandas)], average_all_acs_pandas)
        plt.semilogx(lag_for_plotting_avg_ac, average_all_acs_pandas)
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
        average_dynamic_acs_pandas = tolerant_mean(dynamic_acs_pandas)[0]
        #average_dynamic_acs_pandas = np.average(dynamic_acs_pandas, axis = 0)
        lag_for_plotting_avg_ac = []
        for key,val in chosen_acs_lags_pandas.items():
            if len(val['lag']) == len(average_dynamic_acs_pandas):
                lag_for_plotting_avg_ac  = val['lag'].values
                break
        
        first_key = list(chosen_acs_lags_pandas.keys())[0]
        #print(average_dynamic_acs_pandas)
        plt.figure(figsize = (12,6))
        plt.subplot(121)
        #plt.semilogx(chosen_acs_lags_pandas[first_key]['lag'], average_dynamic_acs_pandas)
        plt.semilogx(lag_for_plotting_avg_ac, average_dynamic_acs_pandas)
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
    


#zz = transparent_image_creator(qimages_250K['q1_all_sp'])                        

# ### 15min indices
# folderpath_index = r"C:\Berkeley\DATA\june23_24\15minutes_seperated_files"
# #file257 = r"250K_1800_warming2.t3pa"
# file_indices = r"15min_chunk_indices.csv"
# full_path_indices = os.path.join(folderpath_index, file_indices)
# iddf = pd.read_csv(full_path_indices,  header = None, skiprows = 0)
# iddf.columns = iddf.iloc[0]
# iddf

# #read257['t'].iloc[second15[0]+1:second15[1]]


# # In[16]:


# first15_ids = str_to_int(iddf['252'][1])
# second15_ids = str_to_int(iddf['252'][2])
# print(first15_ids)
# print(second15_ids)


# In[17]:


# folderpath = r"C:\Berkeley\DATA\june23_24"
# filename = r"252K_warming_1800s2.t3pa"
# #readfile250_chunk = tpx3_2.file_read_chunk(folderpath, filename250, 247337166)
# readfile = tpx3_2.file_read(folderpath, filename)
# first15min = readfile.iloc[0: first15_ids[1] ]
# second15min = readfile.iloc[ second15_ids[0]+1 : second15_ids[1] ]

# first15min_sorted = first15min.sort_values(by = 't', ascending = True)
# second15min_sorted = second15min.sort_values(by = 't', ascending = True)



# In[21]:
