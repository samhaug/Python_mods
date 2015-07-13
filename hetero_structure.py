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
         file = open('/Users/samhaug/Scattering_Profile/VELOCITY_REFERENCE/PREM_1s.csv')
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
         array = np.loadtxt(self.Background)
         radius = array[:,0]
         theta = array[:,1]
         vp_structure = array[:,2]
         vs_structure = array[:,3]
         density = array[:,4]

         self.radius = np.unique(radius)
         self.theta = np.unique(theta)
         self.vp_2D = np.empty((len(self.radius),len(self.theta)))
         self.vs_2D = np.empty((len(self.radius),len(self.theta)))
         self.rho_2D= np.empty((len(self.radius),len(self.theta)))

         for ii in range(0,len(array)):
             r = np.argmin(np.abs(self.radius-array[ii,0]))
             th = np.argmin(np.abs(self.theta-array[ii,1]))
             self.vp_2D[r,th] = array[ii,2]
             self.vs_2D[r,th] = array[ii,3]
             self.rho_2D[r,th] = array[ii,4]

##########kk#####################################################################
   def array_interp1D(self,interp_radius,interp_theta):      
###############################################################################
       '''
       Interpolates density, vp, vs onto different radial domain. This is for
       1D reference values only.
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

##########kk#####################################################################
   def array_interp2D(self,interp_radius,interp_theta):      
###############################################################################
       '''
       Interpolates density, vp, vs onto different radial domain. This is for
       2D models.
       '''
       f_rho = sp.interpolate.interp2d(self.theta,self.radius,self.rho_2D,kind='cubic')
       f_vp = sp.interpolate.interp2d(self.theta,self.radius,self.vp_2D,kind='cubic')
       f_vs = sp.interpolate.interp2d(self.theta,self.radius,self.vs_2D,kind='cubic')
       self.rho_2D = f_rho(interp_theta, interp_radius)
       self.vp_2D = f_vp(interp_theta, interp_radius)
       self.vs_2D = f_vs(interp_theta, interp_radius)
       self.radius = interp_radius
       self.theta = interp_theta
###############################################################################
   def add_hetero_region(self,rmin,rmax,thmin,thmax,vp_new,vs_new,rho_new):
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
       self.rho_2D[min(irmin,irmax):max(irmin,irmax),ithmin:ithmax]=float(rho_new)


###############################################################################
   def add_hetero_point(self,radius,theta,vp_new,vs_new,rho_new,width):
###############################################################################
       '''
       Add square heterogeneity with specified vp, vs, and width at radius and
       angle clockwise from north. this function specifies the most inner, northward
       point of the square.
       '''

       rmin = radius
       rmax = rmin+width
       thmin = theta
       thmax = thmin + (float(width)/float(rmin))*180./np.pi
       self.add_hetero_region(rmin,rmax,thmin,thmax,vp_new,vs_new,rho_new)
           
###############################################################################
   def preview(self,plot='cart'):
###############################################################################
       '''
       Make polar plot of the mantle heterogeneity structure as it currently is
       '''
       
       font = {'family' : 'sans-serif',
               'color' : 'black',
               'weight' : 'normal',
               'size' : 16}

       if plot == 'cart':
           theta,rad = np.meshgrid(self.theta,self.radius)
           fig = plt.figure(num=None, figsize=(9, 12), dpi=80, edgecolor='k')
           ax = fig.add_subplot(211)
           ax.set_ylabel('Radius (km)',fontdict=font)
           ax.set_title('Vs 2D profile',fontdict=font)
           pc1= ax.pcolormesh(theta,rad,self.vs_2D)
           cbar1 = plt.colorbar(pc1)
           cbar1.set_label('Velocity (km/s)',fontdict=font)

           ax2 = fig.add_subplot(212)
           ax2.set_ylabel('Radius (km)',fontdict=font)
           ax2.set_title('Vp 2D profile',fontdict=font)
           ax2.set_xlabel('Degrees from pole',fontdict=font)
           pc2 = ax2.pcolormesh(theta,rad,self.vp_2D)
           cbar2 = plt.colorbar(pc2)
           cbar2.set_label('Velocity (km/s)',fontdict=font)
           plt.show()

       elif plot == 'polar':
           #Set up radial and angular grid points 
           azimuths = np.radians(np.linspace(0,180,num=self.theta.size))
           zeniths = np.linspace(0,6371,num=self.radius.size)
           r, theta = np.meshgrid(zeniths,azimuths)

           fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
           fig.text(0.85,0.5,'Vp Structure')
           CS = ax.contourf(theta,r, np.transpose(np.flipud(self.vp_2D)))
           ax.set_theta_zero_location("N")
           ax.set_theta_direction(-1)
           cbar = plt.colorbar(CS)
           plt.show()
       else:
           print 'Only cartesian and polar plots are supported'

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
       
