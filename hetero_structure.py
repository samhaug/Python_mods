#/usr/bin/env python

import numpy as np
import scipy as sp
from scipy import interpolate
import matplotlib
from matplotlib import pyplot as plt

###############################################################################
class Mantle_Structure(object):
###############################################################################

###############################################################################
   def __init__(self,name,Background):
###############################################################################
      self.Background = Background
      self.name = name
      self.hetero_dict = dict()
      self.num_dict_entries = 0

      if self.Background == 'PREM':
         file = open('/Users/samhaug/Python_mods/VELOCITY_REFERENCE/PREM_1s.csv')
         file = file.read()
         array = file.strip().split('\n')
         array = [ii.split(',') for ii in array]
         self.density = np.zeros(len(array))
         self.radius = np.zeros(len(array))
         self.theta = np.linspace(0,180,num=len(array))
         self.vp_structure = np.zeros(len(array))
         self.vs_structure = np.zeros(len(array))
         for ii in range(0,len(array)):
            self.radius[ii] = array[ii][0]
            self.density[ii] = array[ii][2]
            self.vp_structure[ii] = array[ii][3]
            self.vs_structure[ii] = array[ii][5]
       
         self.vs_2D = np.empty((len(self.radius),len(self.theta)))
         self.vp_2D = np.empty((len(self.radius),len(self.theta)))
         self.rho_2D = np.empty((len(self.radius),len(self.theta)))
         for ii in range(0,len(self.radius)):
            for jj in range(0,len(self.theta)):
               self.vs_2D[ii,jj] = self.vs_structure[ii]
               self.vp_2D[ii,jj] = self.vp_structure[ii]
               self.rho_2D[ii,jj] = self.density[ii]
      else:
         print 'PREM background is only supported'

###############################################################################
   def array_interp(self,interp_radius,interp_theta):      
###############################################################################
       '''
       Interpolates density, vp, vs onto different radial domain.
       '''
       interp_vp = interpolate.interp1d(self.radius,self.vp_structure)
       interp_vs = interpolate.interp1d(self.radius,self.vs_structure)
       interp_rho = interpolate.interp1d(self.radius,self.density)

       self.radius = interp_radius
       self.theta = interp_theta
       self.vp_structure = interp_vp(self.radius)
       self.vs_structure = interp_vs(self.radius)
       self.density = interp_rho(self.radius)

       self.vs_2D = np.empty((len(self.radius),len(self.theta)))
       self.vp_2D = np.empty((len(self.radius),len(self.theta)))
       self.rho_2D = np.empty((len(self.radius),len(self.theta)))
       for ii in range(0,len(self.radius)):
          for jj in range(0,len(self.theta)):
             self.vs_2D[ii,jj] = self.vs_structure[ii]
             self.vp_2D[ii,jj] = self.vp_structure[ii]
             self.rho_2D[ii,jj] = self.density[ii]

###############################################################################
   def add_hetero(self,rmin,rmax,thmin,thmax,vp_new,vs_new,abs=True):
###############################################################################
       '''
       Add vp_new and vs_new heterogeneity within a region bounded by rmin, 
       rmax, thmin, thmax. abs is a boolean that determins if it is a relative 
       or absolute change in velocity. if relative, enter the percent value.
       '''
        
       irmin = (np.abs(self.radius - rmin)).argmin()    
       irmax = (np.abs(self.radius - rmax)).argmin()    
       ithmin = (np.abs(self.theta - thmin)).argmin()    
       ithmax = (np.abs(self.theta - thmax)).argmin()    
       coord_list = [irmin, irmax, ithmin, ithmax, vp_new, vs_new]

       #Add entry to dicitonary that includes heterogeneity info
       self.hetero_dict[self.num_dict_entries] = coord_list
       self.num_dict_entries += 1

       self.vs_2D[min(irmin,irmax):max(irmin,irmax),ithmin:ithmax]=float(vs_new)
       self.vp_2D[min(irmin,irmax):max(irmin,irmax),ithmin:ithmax]=float(vp_new)

           
###############################################################################
   def preview(self,plot='cart'):
###############################################################################
       '''
       Make polar plot of the mantle heterogeneity structure as it currently is
       '''

       if plot == 'cart':
           theta,rad = np.meshgrid(self.theta,self.radius)
           fig = plt.figure()
           ax = fig.add_subplot(211)
           ax.set_ylabel('Radius (km)')
           ax.set_xlabel('Degrees from pole')
           ax.set_title('Vs 2D profile')
           ax.pcolormesh(theta,rad,self.vs_2D)
           ax2 = fig.add_subplot(212)
           ax2.set_ylabel('Radius (km)')
           ax2.set_xlabel('Degrees from pole')
           ax2.set_title('Vp 2D profile')
           ax2.pcolormesh(theta,rad,self.vp_2D)
           plt.show()

       elif plot == 'polar':
           azimuths = (np.linspace(0,180,num=self.theta.size))
           zeniths = np.linspace(0,6371,num=self.radius.size)
           r, theta = np.meshgrid(zeniths,azimuths)
           fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
           ax.contourf(theta,r, np.transpose(np.flipud(self.vp_2D)))
           plt.show()

###############################################################################
   def output_1D_sph(self):
###############################################################################
       '''
       Returns .sph file for the Mantle_Structure class. This can be used in the
       inparam_hetero file of AxiSEM.
       '''

       file = open(self.name+'.sph','w+')
       file.write(str(self.theta.size*self.radius.size)+'\n')
       for ii in range(0,len(self.radius)):
           for jj in range(0,len(self.theta)):
               file.write(str(self.radius[ii])+' '+ \
                          str(np.round(self.theta[jj],decimals=2))+' '+ \
                          str(self.vp_structure[ii]*1000.)+' '+ \
                          str(self.vs_structure[ii]*1000.)+' '+ \
                          str(self.density[ii]*1000.)+'\n')
       file.close()

###############################################################################
   def output_2D_sph(self):
###############################################################################
       '''
       Returns 2D .sph file for the Mantle_Structure class. This can be used in the
       inparam_hetero file of AxiSEM.
       '''

       self.radius_2D  = np.array(np.tile(self.radius,(self.theta.size,1)).transpose())
       self.theta_2D   = np.tile(self.theta,(self.radius.size,1))
       
       self.reshape_radius = self.radius_2D.reshape(self.radius_2D.size,1)
       self.reshape_theta  = self.theta_2D.reshape(self.theta_2D.size,1)
       self.reshape_vp  = self.vp_2D.reshape(self.vp_2D.size,1)
       self.reshape_vs  = self.vs_2D.reshape(self.vp_2D.size,1)
       self.reshape_rho = self.rho_2D.reshape(self.vp_2D.size,1)
       
       file = open(self.name+'_2D.sph','w+')
       file.write(str(self.reshape_radius.size)+'\n')

       for ii in range(0,len(self.reshape_radius)):
           file.write(str(np.around(self.reshape_radius[ii][0],decimals=3))+' '+ \
                   str(np.around(self.reshape_theta[ii][0],decimals=3))+' '+ \
                   str(np.around(self.reshape_vp[ii][0],decimals=3))+' '+ \
                   str(np.around(self.reshape_vs[ii][0],decimals=3))+' '+ \
                   str(np.around(self.reshape_rho[ii][0],decimals=3))+'\n')

       file.close()
       
