import numpy as np


"""
notes on the fluid state variables:
    here we suppose, that compressible fluid is represented by density field (dens), velocity field (vel1,vel2,vel3) 
    and pressure field (pres). additionally we have a conservative fluid state with 
    mass, momentum(1,2,3) and total energy, all of them are given for the unit volume
    
The "boundMark" variable is a boundary marker. it marks each boundary (by default the boundary marker is set to 100)
    100 - non-reflecting boundary condition (zero gradient)
    101 - wall bounary condition (normal velocity is set to zero)
    300 - periodic boundaries (even or odd indexes have to the same in this case)
    
@author: mrkondratyev
"""

class FluidState:
    def __init__(self, grid):

        # Fluid state at each grid cell including ghost cells
        self.dens = np.zeros(grid.grid_shape, dtype=np.double)
        self.vel1 = np.zeros(grid.grid_shape, dtype=np.double)
        self.vel2 = np.zeros(grid.grid_shape, dtype=np.double)
        self.vel3 = np.zeros(grid.grid_shape, dtype=np.double)
        self.pres = np.zeros(grid.grid_shape, dtype=np.double)
        
        # Conservative variables inside the real cells only (!)
        self.mass = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.mom1 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.mom2 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.mom3 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.etot = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        
        
        #boundary marker - by default we set transmissive boundaries
        self.boundMark = np.zeros(4, dtype=np.int32)
        self.boundMark[:] = 100
        
        #user-defined source terms (e.g. gravity acceleration and so on)
        self.F1 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.F2 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)

        
        