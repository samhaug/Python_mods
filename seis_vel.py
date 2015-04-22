#!/usr/bin/env python

'''
seis_vel is a collection of functions for preparing a seismic velocity structure
to be used as an input file for the AxiSEM program.

Written by Samuel Haugland and Ross Maguire
'''

import numpy as np
import math
import obspy
from obspy.taup import TauPyModel
import matplotlib.pyplot as plt


###############################################################################
def get_base_values(datfile):
###############################################################################
   '''
   Get temperature and pressure step values from cpl file. This function must 
   be run before the other lookup functions in this module can.

   Parameters
   __________
   
   datfile : file name
      A cpl. file wit

   Returns
   _______
   
   T0 : float
      Initial temperature value

   P0 : float
      Initial pressure value

   dT : float
      Temperature step size

   dP : float
      Pressure step size

   nT : int
      Number of temperature steps

   nP : int
      Number of temperature steps
   '''

   file = open(datfile)
   file.seek(13)
   twoline = file.readline()
   nP = int(twoline.strip().split()[0])
   nT = int(twoline.strip().split()[1])
   P0 = float(twoline.strip().split()[2])
   T0 = float(twoline.strip().split()[3])
   dP = float(twoline.strip().split()[4])
   file.seek(69)
   threeline = file.readline()
   dT = float(threeline.strip())
   file.close()
   
   return T0, P0, dT, dP, nT, nP
   
###############################################################################
def array_generator(vp_file,vs_file,rho_file):
###############################################################################
   '''

   Generate P-T lookup tables for vp, vs, and rho.
   
   Parameters
   __________
   
   vp_file : file name 
      File path containing vp information. File name is cpl."..."

   vs_file : file name 
      File path containing vs information. File name is cpl."..."

   rho_file : file name 
      File path containing rho information. File name is cpl."..."

   Returns
   _______

   vp_array : numpy array
      Lookup table of vp values

   vs_array : numpy array
      Lookup table of vs values

   density_array : numpy array
      Lookup table of density values
   '''
   
   vp_pyrolite = np.loadtxt(vp_file,skiprows=4)
   vs_pyrolite = np.loadtxt(vs_file,skiprows=4)
   density_pyrolite = np.loadtxt(rho_file,skiprows=4)

   file = open(vp_file)
   file.seek(13)
   twoline = file.readline()
   nP = int(twoline.strip().split()[0])
   nT = int(twoline.strip().split()[1])
   P0 = float(twoline.strip().split()[2])
   T0 = float(twoline.strip().split()[3])
   dP = float(twoline.strip().split()[4])
   

   vp_array = np.zeros((nP,nT))
   vs_array = np.zeros((nP,nT))
   density_array = np.zeros((nP,nT))
   for ii in range(0,nP):
      for jj in range(0,nT):
         global_index = ii*nT+jj
         vp_array[ii,jj] = vp_pyrolite[global_index]
         vs_array[ii,jj] = vs_pyrolite[global_index]
         density_array[ii,jj] = density_pyrolite[global_index]
   
   return vp_array, vs_array, density_array 

###############################################################################
def find_vp(T, P, vp_array,P0,T0,dP,dT):
###############################################################################
   '''
   Lookup vp value for given T-P conditions
   
   Parameters
   __________
  
   T : float
      Temperature in C 

   P : float
      Pressure in GPa

   vp_array : numpy array
      vp lookup table
   
   datfile : file
      Any cpl."..." file. The first few lines will be read and stripped for 
      P-T information. This information will be the same for all cpl files in 
      the same directory

   Returns
   _______

   vp : float
      Seismic P-wave velocity   
   '''

   #T = T+273.0      
   P = P*10000

   index_P = int((P)/dP) - int((P0)/dP)
   index_T = int((T)/dT) - int((T0)/dT)

   if(index_T < 0):
      index_T = 0

   if(index_P < 0):
      index_P = 0

   vp = vp_array[index_P, index_T]

   return vp

###############################################################################
def find_vs(T, P, vs_array, P0, T0, dP, dT):
###############################################################################
   '''
   SEE HELP DOCUMENTATION FOR find_vp. THIS IS SIMILAR EXCEPT vp=vs
   '''

   #T = T+273.0      
   P = P*10000

   index_P = int((P)/dP) - int((P0)/dP)
   index_T = int((T)/dT) - int((T0)/dT)

   if(index_T < 0):
      index_T = 0

   if(index_P < 0):
      index_P = 0

   vs = vs_array[index_P, index_T]

   return vs

###############################################################################
def find_density(T, P, density_array, P0 ,T0 ,dP , dT):
###############################################################################
   '''
   SEE HELP DOCUMENTATION FOR find_vp. THIS IS SIMILAR EXCEPT vp=density
   '''
 
   #T = T+273.0      
   P = P*10000

   index_P = int((P)/dP) - int((P0)/dP)
   index_T = int((T)/dT) - int((T0)/dT)

   if(index_T < 0):
      index_T = 0

   if(index_P < 0):
      index_P = 0

   density = density_array[index_P, index_T]

   return density

###############################################################################
def xy_2_pressure(x, y):
###############################################################################
   '''
   Determines lithostatic pressure
   
   Parameters
   __________

   x : float
      x coordinates in mantle

   y : float
      y coordinates in mantle
   
   Returns
   _______
   
   pressure : float 
      Lithostatic pressure in GPa 

   radius : float
      radius

   theta : float
      colatitude in radians
   '''

   earth_radius = 6371000.0
   density = 3300
   g = 9.81
  
   radius = np.sqrt(pow(x*1000,2)+pow(y*1000,2))
   pressure = density*g*(earth_radius-radius)/pow(10,9)

   if y > 0:
      theta = (90. - math.atan2(y,x)*180/np.pi)*np.pi/180.
   else:
      theta = (-(math.atan2(y,x)*180/np.pi) +90.)*np.pi/180.

   #reshaped = np.array([np.transpose(pressure),np.transpose(radius),np.transpose(theta)])

   return pressure, radius, theta
   
###############################################################################
def xy_2_rt(x, y):
###############################################################################

   radius = np.sqrt(pow(x*1000,2)+pow(y*1000,2))
      
   if y > 0:
      theta = (90. - math.atan2(y,x)*180/np.pi)*np.pi/180.
   else:
      theta = (-(math.atan2(y,x)*180/np.pi) +90.)*np.pi/180.

   return radius, theta

###############################################################################
def plot_expected_arrivals(source_depth, distance, ref_model, phase_list, seislist):
###############################################################################
   '''
   Plot expected arrival times based on seismic reference model ref_model
   ref_model must be a string. source depth in km. distance in degrees.
   phase_list is list of phase names you want to plot. seis is a numpy array for 
   a seismic trace.
   See obpy taup manual
   '''

   model = TauPyModel(model=ref_model)
   arrivals = model.get_travel_times(source_depth_in_km=source_depth, \
           distance_in_degree=distance)

   seis_number = len(seislist)
   #seis = seislist
   #seis_array = np.loadtxt(seis) 
   #max_value = seis_array[:,1].max()

   arrive_list = str(arrivals).strip().split('\n')
   arrive_array = [ii.split() for ii in arrive_list]
   phase_dict = dict()
   for ii in range(1,len(arrive_array)):
      phase_dict[arrive_array[ii][0]] = float(arrive_array[ii][4])

   print 'Phases Available: \n'
   print phase_dict.keys()

   for jj in range(0,len(seislist)):
      plt.subplot(seis_number,1,(jj+1))
      seis_array = np.loadtxt(seislist[jj])
      max_value = seis_array[:,1].max()*0.9

      #Remove y lables
      ax = plt.gca()
      ax.set_yticklabels([])
      #Label plot
      plt.title(seislist[jj])
      plt.xlabel('Seconds')
      #change limits
      plt.xlim(seis_array[0,0],seis_array[len(seis_array)-1,0])
      
      for ii in phase_list:
         plt.axvline(x=phase_dict[ii], ymin=-1.0, ymax = 1.0, \
                linewidth=2, color='gold')
         plt.text(phase_dict[ii],max_value,ii)
         plt.text(0,max_value,'Vel. model '+ref_model)
      plt.plot(seis_array[:,0],seis_array[:,1])

   plt.show()

###############################################################################
def taup_array_maker(source_depth, dist_range, ref_model, phase_list, seisfile):
###############################################################################
   '''
   Finds ray paths for a certain phase over a range of seismograms. Arranges them
   in a dictionary.

   Parameters
   __________
   

   Returns
   _______

   '''
   from obspy.taup import TauPyModel
   model = TauPyModel(model = ref_model)
   
   degree_array = np.arange(dist_range[0],dist_range[1]+1,1)
   arrival_list = list() 

   for ii in degree_array:
       arrival = model.get_ray_paths(source_depth,ii)
       for jj in range(0,len(arrival)):
          if arrival[jj] in phase_list:
