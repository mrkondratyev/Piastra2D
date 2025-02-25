# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:04:19 2023


The main code -- 
grid construction, 
initial data filling, 
simulation control and visualization live here 


@author: mrkondratyev
"""

from grid_setup import Grid
from init_cond_fluid import *
from fluid_state import FluidState
from CFL_condition import CFLcondition_fluid
from aux_routines import auxData
from one_step_fluid import oneStep_fluid_RK
import matplotlib.pyplot as plt
import numpy as np
import time
from visualization import visual

#here we introduce the grid cell numbers in each direction + the number of ghost cells
Nx1 = 64
Nx2 = 64
Ngc = 3

#here we initialize the grid
grid = Grid(Nx1, Nx2, Ngc)

#coordinate range in each direction, by default x and y are in range [0..1]
x1ini, x1fin = 0.0, 1.0
x2ini, x2fin = 0.0, 1.0

#filling the grid arrays with grid data (by now it is only uniform Cartesian grid)
grid.uniCartGrid(x1ini, x1fin, x2ini, x2fin)

#obtaining auxilary data (timestep, final time and so on)
aux = auxData(grid)

#initialize fluid state
fluid = FluidState(grid)

#fill the fluid state arrays with initial data
#see "init_cond.py" for different examples/tests
###############################################################
fluid, aux, eos = init_cond_KH_inst_2D(grid, fluid, aux)
#fluid, aux, eos =  init_cond_Sod_cart_1D(grid, fluid, aux)
###############################################################

print("grid resolution = ", grid.Nx1, grid.Nx2)

#here we adjust the solver parameters and print them
aux.rec_type = 'PLM'
aux.flux_type = 'LLF'
aux.RK_order = 'RK2'
print("reconstruction type = ", aux.rec_type)
print("Riemann flux = ", aux.flux_type)
print("Temporal integration = ", aux.RK_order)

#print final phys time 
print("final phys time = ", aux.Tfin)    


#set the start timer to check the elapsed time 
start_time1 = time.time() 

print("START OF SIMULATION")

#cycle over time 
i_time = 0
while aux.time < aux.Tfin:
       
    #current timestep
    i_time = i_time + 1
    
    #calculate the avaliable timestep according to Courant-Friedrichs-Lewy condition
    dt = CFLcondition_fluid(grid, fluid, eos, aux.CFL)
    dt = min(dt, aux.Tfin - aux.time)
    
    #fluid state variables update 
    fluid = oneStep_fluid_RK(grid, fluid, eos, dt, aux.rec_type, aux.flux_type, aux.RK_order)
    
    #"real time" output (animated)
    aux.time = aux.time + dt
    if (i_time%25 == 0) or (aux.Tfin - aux.time) < 1e-12:
        plt.clf()
        rhomin = np.min(fluid.dens[Ngc:-Ngc,Ngc:-Ngc])
        rhomax = np.max(fluid.dens[Ngc:-Ngc,Ngc:-Ngc])
        print("phys time = ", aux.time)
        print('num of timesteps = ', i_time)
        plt.cla()
        if (grid.Nx2 == 1): 
            plt.plot(grid.cx1[Ngc:-Ngc,Ngc], fluid.dens[Ngc:-Ngc,Ngc])
            
        if (grid.Nx1 == 1):
            plt.plot(grid.cx2[Ngc,Ngc:-Ngc], fluid.dens[Ngc,Ngc:-Ngc])
        
        if (grid.Nx1 != 1 & grid.Nx2 != 1):
            plt.imshow(fluid.dens[grid.Ngc:-grid.Ngc, grid.Ngc:-grid.Ngc], cmap='jet')
            
            plt.clim(rhomin, rhomax)
            #plt.clim(1.0, 2.0)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)	
            ax.set_aspect('equal')	
            plt.pause(0.03)
 
#print final physical time
print("final phys time = ", aux.time)    

print("END OF SIMULATION")
##calculate and the elapsed time of the simulation
end_time1 = time.time()
print("time of simulation = ", end_time1 - start_time1, " secs")
  
# Show the plot
plt.show()
if (grid.Nx2 == 1): 
    plt.plot(grid.cx1[Ngc:-Ngc,Ngc], fluid.dens[Ngc:-Ngc,Ngc])

if (grid.Nx1 == 1):
    plt.plot(grid.cx2[Ngc,Ngc:-Ngc], fluid.dens[Ngc,Ngc:-Ngc])

#plt.imshow(fluid.dens, extent=(grid.cx1.min(), grid.cx1.max(), grid.cx2.min(), grid.cx2.max()), origin='lower', cmap='viridis', interpolation='nearest', aspect='auto')

# Add labels and a colorbar
#plt.colorbar(label='Colorbar Label')
#plt.xlabel('X Label')
#plt.ylabel('Y Label')
#plt.title('2D Plot of Data')

