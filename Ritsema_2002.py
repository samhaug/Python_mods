#usr/bin/env python
'''
Ritsema_2002.py is a module for computing relevant values found in the following 
paper: 

"Constaints onf the correlation of P- and S-wave velocity heterogeneity
in the mantle form P, PP, PPP , and PKPab traveltimes." J. Ritsema, H. van
Heijst, GJI 2002

Also other papers of importance.

Made by Samuel Haugland, June 2015
'''

from matplotlib import pyplot as plt
import obspy
from obspy.taup import TauPyModel
import obspy.signal.filter
import obspy.signal
import numpy as np
model = TauPyModel(model="iasp91")
from matplotlib import colors, ticker, cm
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap

###############################################################################
def depth_distance(seis_trace):
###############################################################################
    '''
    Use to quickly find source depth and receiver distance from xh trace file.
    '''
    source_depth = seis_trace.stats.xh['source_depth_in_km']
    receiver_distance_in_deg = seis_trace.stats.xh['receiver_latitude']
    if receiver_distance_in_deg > 0:
        receiver_distance_in_deg = 90.-receiver_distance_in_deg
    elif receiver_distance_in_deg < 0:
        receiver_distance_in_deg = 90.+abs(receiver_distance_in_deg)

    return source_depth, receiver_distance_in_deg

    

###############################################################################
def F_1_F_2(data_seis, synth_seis):
###############################################################################
    '''
    Calculate F_1 and F_2  as shown in equation (1) and (2)in the paper.
    
    PARAMETERS
    __________
    
    data_seis: obspy.core.trace.Trace object.
    
    synth_seis: obspy.core.trace.Trace object.
    
    RETURNS
    _______
    
    F_tuple: (F_1,F_2) tuple
    '''
    # find F_1
    cross_correllate = scipy.signal.correlate(synth_seis.data,data_seis.data)
    offset = numpy.argmin(max(cross_correllate)-cross_correllate)
    synth_seis_shift = numpy.roll(synth_seis.data,(len(synth_seis)-offset-1))
     
    Numerator = numpy.sum((data_seis.data-synth_seis_shift)**2)
    Denomenator = numpy.sum(data_seis.data**2)

    F_1 = Numerator/Denomenator

    # find F_2

    A1 = numpy.linalg.lstsq(synth_seis_shift[:,None],data_seis.data)[0][0]
    A2 = 1./numpy.linalg.lstsq(data_seis.data[:,None],synth_seis_shift)[0][0]

    F_2 = min((A1,A2))/max((A1,A2))

    return (F_1,F_2)

###############################################################################
def Earle_STN(seis_trace):
###############################################################################
    '''
    Calculate signal-to-noise (STN) as described in:
    
    "Distribution of fine-scale mantle heterogeneity from observations of Pdiff
    coda" Earle & Shearer 2001.
    
    PARAMETERS
    __________
    
    seis_trace: obspy.core.trace.Trace object.

    RETURNS  
    __________

    STN: float
    
    
    ''' 

    source_depth = seis_trace.stats.xh['source_depth_in_km']
    receiver_distance_in_deg = seis_trace.stats.xh['receiver_latitude']
    if receiver_distance_in_deg > 0:
        receiver_distance_in_deg = 90.-receiver_distance_in_deg
    elif receiver_distance_in_deg < 0:
        receiver_distance_in_deg = 90.+abs(receiver_distance_in_deg)
        
    arrivals = model.get_travel_times(source_depth_in_km=source_depth,
           distance_in_degree = receiver_distance_in_deg, phase_list = ['PP','Pdiff'])

    Pdiff_time = arrivals[0].time
    PP_time = arrivals[1].time

    seis_trace.filter('bandpass',freqmin=0.5,freqmax=2.5)

    numerator = seis_trace.slice(seis_trace.stats.starttime+PP_time,
            seis_trace.stats.starttime+PP_time+40.).max()
    denominator = seis_trace.slice(seis_trace.stats.starttime+Pdiff_time-120.,
            seis_trace.stats.starttime+Pdiff_time-50.).max()

    STN = numerator/denominator

    return STN


###############################################################################
def Earle_amplitude_stack(seis_stream):
###############################################################################

    '''
    Stack traces and normalize envelopes like figure 2 of Earle & Shearer 2001. 
    This function assumes that all traces in seis_stream are from the same
    distance from the source.
    
    PARAMETERS
    __________
    
    seis_stream: obspy.core.stream.Stream object.
    
    RETURNS
    _______

    ''' 
    seis_list = []
    seis_dict = {}
    distance_list = [95.,100.,105.,110.,115.,120.,125.,130.]
    for tr in seis_stream:
        source_depth, receiver_distance_in_degree = depth_distance(tr)
        if receiver_distance_in_degree in distance_list:
            print receiver_distance_in_degree
            arrivals = model.get_travel_times(source_depth_in_km=source_depth,
            distance_in_degree = receiver_distance_in_degree, phase_list = ['PP','Pdiff'])

            PP_time = arrivals[0].time
           # Pdiff_time = arrivals[1].time
           # print Pdiff_time
            print PP_time

        # Find PP amplitude and normalize trace to that amplitude
            PP_amplitude = abs(tr.slice(tr.stats.starttime+PP_time-2.,
                tr.stats.starttime+PP_time+2.).max())
            tr.data = tr.data/PP_amplitude
        # Window seismic trace to include Pdiff and PP phase. See figure 2 
            tr = tr.slice(tr.stats.starttime+PP_time-400,tr.stats.starttime+PP_time+100)
            envelope = obspy.signal.filter.envelope(tr.data)+receiver_distance_in_degree/10.
            seis_dict[receiver_distance_in_degree] = [envelope]
            seis_list += [envelope]

    seis_array = np.array(seis_list)
    
    for ii in range(0,seis_array.shape[0]):
        plt.plot(seis_array[ii,:])
    plt.show()

    return seis_array
        

###############################################################################
def Shearer_PKPdf_colorbar(seis_stream,event_depth):
###############################################################################
    '''
    Produce colorbar of PKPdf scatterers from Mancinelli & Shearer 2011 
    PARAMETERS
    __________
    
    seis_stream: obspy.core.stream.Stream object.
    '''

    shearer_stream = obspy.core.stream.Stream()
    envelope_dict = {}
    for tr in seis_stream:
        source_depth, receiver_distance_in_degree = depth_distance(tr)
        if 120. <= receiver_distance_in_degree <= 145.:
            tr.filter('bandpass',freqmin=0.7,freqmax=2.5)
            tr2 = tr.copy()
            degrees = np.round(tr.stats.distance/111000.)
            arrivals = model.get_travel_times(source_depth_in_km=event_depth,
            distance_in_degree = degrees, phase_list = ['PKiKP'])

# Find the PKiKP arrival by looking for a maximum amplitude in small window around
# expected IASP91 arrival.
            PKiKP_iasp91_time = arrivals[0].time
            tr_avg_noise = tr.slice(tr.stats.starttime+PKiKP_iasp91_time-25,
                tr.stats.starttime+PKiKP_iasp91_time+20).data.mean()
            tr = tr.slice(tr.stats.starttime+PKiKP_iasp91_time-0.1,
                tr.stats.starttime+PKiKP_iasp91_time+0.1)
            max_arrival = tr.stats.starttime+np.argmin(abs(tr.max()-tr))/tr.stats.sampling_rate
            
            absolute_arrival =  max_arrival-obspy.core.UTCDateTime(0)
            tr_abs = tr2.slice(tr2.stats.starttime+absolute_arrival-30.,
                 tr2.stats.starttime+absolute_arrival-20).data.mean()
            tr2.data = tr2.data-abs(tr_abs)
            #tr.plot()
            if receiver_distance_in_degree >= 142.:
                tr2 = tr2.slice(tr2.stats.starttime+PKiKP_iasp91_time-30.,
                    tr2.stats.starttime+PKiKP_iasp91_time+20)

            else:
                tr2 = tr2.slice(tr2.stats.starttime+absolute_arrival-30.,
                    tr2.stats.starttime+absolute_arrival+20)

            #print obspy.core.UTCDateTime(PKiKP_seconds)
            #print "iasp91   "+str(obspy.core.UTCDateTime(PKiKP_iasp91_time))
            #print "abs      "+str(PKiKP_abs_time) 

            envelope_dict[degrees] = abs(obspy.signal.filter.envelope(tr2.data))
            shearer_stream += tr2

    array_size = envelope_dict[120.].size
    shearer_array = np.zeros(array_size)[:,None]
    for idx,ii in enumerate(range(120,146)):
        shearer_array = np.hstack((shearer_array,envelope_dict[ii][0:6469][:,None]))
    
# Normalize shearer_array
    for ii in range(0,shearer_array.shape[1]):
        shearer_array[:,ii] = shearer_array[:,ii]/shearer_array[:,ii].max()
    test=hex_2_rgb()
    shearer_array = np.delete(shearer_array,0,1)
    shearer_array = pow(shearer_array,2)
    degrees = np.arange(120,146)
    seconds = np.linspace(-25,25,num=6469)
    DEG, SEC = np.meshgrid(degrees, seconds)
    plt.pcolor(DEG,SEC,shearer_array,norm=LogNorm(vmin=shearer_array.min()*pow(10,6),
           vmax=shearer_array.max()-shearer_array.max()/2.),cmap=test)
    plt.xlabel('Distance from source (deg)')
    plt.xlabel('Distance from source (deg)')
    plt.ylim(-25,25)
    plt.colorbar()
    plt.show()

    plt.contourf(DEG,SEC,shearer_array,norm=LogNorm(vmin=shearer_array.min()*pow(10,6),
           vmax=shearer_array.max()-shearer_array.max()/2.),cmap=test)
    plt.xlabel('Distance from source (deg)')
    plt.xlabel('Distance from source (deg)')
    plt.ylim(-25,25)
    plt.colorbar()
    plt.show()
    #plt.pcolor(DEG, SEC, shearer_array, norm=LogNorm(vmin=shearer_array.min(),
    #     vmax=shearer_array.max()),cmap='gnuplot')
    #plt.clim(0,0.0005)
    #plt.ylim(-25,25)
    #plt.xlim(0,shearer_array.shape[1])
    #plt.xlabel('Distance from source (deg)')
    #plt.xlabel('Distance from source (deg)')
    #plt.show()
    
    return shearer_stream, envelope_dict, shearer_array
    

###############################################################################
def hex_2_rgb():
###############################################################################
    '''
    Produce colorbar of PKPdf scatterers from Mancinelli & Shearer 2011 
    PARAMETERS
    __________
    
    seis_stream: obspy.core.stream.Stream object.
    '''
    def hex_to_rgb(value):
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    hex_list = ['#A40844',
                '#BA1f49',
                '#CF384E',
                '#E3534A',
                '#F26A44',
                '#F98F52',
                '#FCA85F',
                '#FDC777',
                '#FEDE88',
                '#FEF0A6',
                '#FFFEBD',
                '#F2FAAB',
                '#B7C27A',
                '#C7E99E',
                '#B0DEA3',
                '#86CFA4',
                '#6CC4A5',
                '#4DA5B0',
                '#3489BC',
                '#496AAF',
                '#5853A4']
    
    return  matplotlib.colors.ListedColormap(hex_list[::-1],name='hey')
    #return matplotlib.colors.ListedColormap(colormap.colors[::,-1])





