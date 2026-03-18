import meep as mp
import meep.adjoint as mpa
import autograd.numpy as npa
from autograd import tensor_jacobian_product
import numpy as np
import nlopt
import time
import os
import glob
import re
from pathlib import Path
from tqdm import tqdm

#Constants

# ----- Parameters -----
WAVEGUIDE_WIDTH = 0.5       # width of input/output waveguides (µm)
DESIGN_REGION_WIDTH = 2.5   # width of the MMI design region
DESIGN_REGION_HEIGHT = 2.5  # height of the MMI design region
ARM_SEPARATION = 1.0        # vertical separation between the two output arms
WAVEGUIDE_LENGTH = 0.5      # length of straight waveguide before/after design region
PML_SIZE = 1.0
RESOLUTION = 30             # pixels/µm 

SaveParameters = [WAVEGUIDE_WIDTH, DESIGN_REGION_WIDTH, DESIGN_REGION_HEIGHT, ARM_SEPARATION, WAVEGUIDE_LENGTH, PML_SIZE,RESOLUTION]


# ADJOINT OPTIMIZATION PARAMETERS
mes = "This is an optimization for a 90:10 splitter resolution 30, minimum feature size is 50nm" #Message to be saved with the file
power1 = 0.9
splitratio = power1/(1-power1)
# Hyperparameters 
cur_beta = 1
beta_scale = 1.2
num_betas = 25
update_factor = 100


total_evals = num_betas * update_factor

CHECKMAS = mp.am_master() 

if(CHECKMAS):
    evalarr = [] # creation of loss array
# Materials (common Si / SiO2 for 1.55 µm)
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)

mp.verbosity(0) #disable this if you want more debug statements from meep, this gives the smallest 




#Function to save file in opts with next index, gives string.
def next_opt_filename(folder="./opts/", base="optresult", ext=".npz", justfile=False): 
    folder = Path(folder)
    pattern = re.compile(rf"{re.escape(base)}\((\d+)\){re.escape(ext)}")

    indices = []
    for f in folder.iterdir():
        m = pattern.fullmatch(f.name)
        if m:
            indices.append(int(m.group(1)))

    next_i = max(indices, default=-1) + 1
    if (justfile):
        return f"{base}({next_i})"
    return folder / f"{base}({next_i}){ext}"





# ----- Simulation cell -----
# Total cell: PML + waveguide + design region + waveguide + PML in x;
#             PML + design_region_height + margin in y.
Sx = 2 * PML_SIZE + 2 * WAVEGUIDE_LENGTH + DESIGN_REGION_WIDTH
Sy = 2 * PML_SIZE + DESIGN_REGION_HEIGHT + 0.5
cell_size = mp.Vector3(Sx, Sy)
pml_layers = [mp.PML(PML_SIZE)]


# ----- Design region (the box we will optimize) -----
# Resolution in the design region is often finer than the simulation.
design_region_resolution = int(5 * RESOLUTION)
Nx = int(design_region_resolution * DESIGN_REGION_WIDTH) + 1
Ny = int(design_region_resolution * DESIGN_REGION_HEIGHT) + 1

# MaterialGrid: interpolates between SiO2 (weight 0) and Si (weight 1).
# Initial weights = 0.5 (gray); we'll optimize them later.
design_variables = mp.MaterialGrid(
    mp.Vector3(Nx, Ny),
    SiO2,
    Si,
    grid_type="U_MEAN",
)
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(),
        size=mp.Vector3(DESIGN_REGION_WIDTH, DESIGN_REGION_HEIGHT, 0),
    ),
)

# ----- Geometry: input waveguide, two output arms, design region -----
# Left: one input waveguide. Right: two output waveguides (top and bottom).
geometry = [
    # Left waveguide (input)
    mp.Block(
        center=mp.Vector3(x=-Sx / 4),
        material=Si,
        size=mp.Vector3(Sx / 2 + 1, WAVEGUIDE_WIDTH, 0),
    ),
    # Top right output arm
    mp.Block(
        center=mp.Vector3(x=Sx / 4, y=ARM_SEPARATION / 2),
        material=Si,
        size=mp.Vector3(Sx / 2 + 1, WAVEGUIDE_WIDTH, 0),
    ),
    # Bottom right output arm
    mp.Block(
        center=mp.Vector3(x=Sx / 4, y=-ARM_SEPARATION / 2),
        material=Si,
        size=mp.Vector3(Sx / 2 + 1, WAVEGUIDE_WIDTH, 0),
    ),
    # Design region (optimizable)
    mp.Block(
        center=design_region.center,
        size=design_region.size,
        material=design_variables,
    ),
]

## Setting sources


fcen = 1 / 1.56
fwidth = 0.2 * fcen
source_center = mp.Vector3(-Sx / 2 + PML_SIZE + WAVEGUIDE_LENGTH / 3, 0, 0)
source_size = mp.Vector3(0, 2, 0)
kpoint = mp.Vector3(1, 0, 0)
source = [
    mp.EigenModeSource(
        mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        eig_band=1, direction=mp.NO_DIRECTION, eig_kpoint=kpoint,
        size=source_size, center=source_center,
    )
]

# Multiple frequencies for broadband design
frequencies = 1 / np.linspace(1.5, 1.6, 5)


sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=SiO2,
    resolution=RESOLUTION,
)


#Setting output directory of Sim
sim.use_output_directory("outputs")



mode = 1

#setting the geometry volume region for the mode monitors

TE0_vol = mp.Volume( center=mp.Vector3(x=-Sx / 2 + PML_SIZE + 2 * WAVEGUIDE_LENGTH / 3), size=mp.Vector3(y=1.5),)
TEtop_vol = mp.Volume( center=mp.Vector3(Sx / 2 - PML_SIZE - 2 * WAVEGUIDE_LENGTH / 3, ARM_SEPARATION / 2, 0), size=mp.Vector3(y=ARM_SEPARATION),)
TEbot_vol = mp.Volume( center=mp.Vector3(Sx / 2 - PML_SIZE - 2 * WAVEGUIDE_LENGTH / 3, -ARM_SEPARATION / 2, 0), size=mp.Vector3(y=ARM_SEPARATION),)

vol_list=[TE0_vol, TEtop_vol, TEbot_vol]
TE0 = mpa.EigenmodeCoefficient(
    sim,
    TE0_vol,
    mode,
)
TE_top = mpa.EigenmodeCoefficient(
    sim,
    TEtop_vol,
    mode,
)
TE_bottom = mpa.EigenmodeCoefficient(
    sim,
    TEbot_vol,
    mode,
)
ob_list = [TE0, TE_top, TE_bottom]

#Objective function to rate the simulation.
def J(source, top, bottom):
    t = npa.abs(top / source)
    b = npa.abs(bottom / source)
    z = 10e-3
    power = t**2 + b**2 - npa.sqrt((t-splitratio*b)**2 + z) 
    return npa.mean(power)


opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J,
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-5,
)

minimum_length = 0.05
eta_e = 0.55
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
eta_i = 0.5

def mapping(x, eta, beta):
    """Filter -> tanh projection -> optional symmetry. Returns flattened weights."""
    filtered = mpa.conic_filter(
        x, filter_radius,
        DESIGN_REGION_WIDTH, DESIGN_REGION_HEIGHT,
        design_region_resolution,
    )
    projected = mpa.tanh_projection(filtered, beta, eta)
    # Enforce up-down symmetry (optional for symmetric splitter)
    # projected = (npa.fliplr(projected) + projected) / 2 commented out for lr splitratio
    return projected.flatten()



# ----- Boundary masks (waveguide ports stay Si, outer border SiO2) -----

x_g = np.linspace(-DESIGN_REGION_WIDTH / 2, DESIGN_REGION_WIDTH / 2, Nx)
y_g = np.linspace(-DESIGN_REGION_HEIGHT / 2, DESIGN_REGION_HEIGHT / 2, Ny)
X_g, Y_g = np.meshgrid(x_g, y_g, sparse=True, indexing="ij")

left_wg_mask = (X_g == -DESIGN_REGION_WIDTH / 2) & (np.abs(Y_g) <= WAVEGUIDE_WIDTH / 2)
top_right_wg_mask = (X_g == DESIGN_REGION_WIDTH / 2) & (
    np.abs(Y_g + ARM_SEPARATION / 2) <= WAVEGUIDE_WIDTH / 2
)
bottom_right_wg_mask = (X_g == DESIGN_REGION_WIDTH / 2) & (
    np.abs(Y_g - ARM_SEPARATION / 2) <= WAVEGUIDE_WIDTH / 2
)
Si_mask = left_wg_mask | top_right_wg_mask | bottom_right_wg_mask

border_mask = (
    (X_g == -DESIGN_REGION_WIDTH / 2) | (X_g == DESIGN_REGION_WIDTH / 2)
    | (Y_g == -DESIGN_REGION_HEIGHT / 2) | (Y_g == DESIGN_REGION_HEIGHT / 2)
)
SiO2_mask = border_mask.copy()
SiO2_mask[Si_mask] = False




cur_iter = [0]
pbar = None
if(CHECKMAS):
    pbar = tqdm(total=total_evals, desc ="Optimizing")

def f(v, gradient, cur_beta):
    x_mapped = mapping(v, eta_i, cur_beta)
    f0, dJ_du = opt([x_mapped])
    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(
            v, eta_i, cur_beta, np.sum(dJ_du, axis=1)
        )
    cur_iter[0] += 1
    if(CHECKMAS):
        evalarr.append(np.real(f0))
        pbar.update(1)
    return np.real(f0)

n = Nx * Ny
reuse = False
if(not reuse):
    x = np.ones(n) * 0.5
    x[Si_mask.flatten()] = 1.0
    x[SiO2_mask.flatten()] = 0.0

lb = np.zeros(n)
lb[Si_mask.flatten()] = 1.0
ub = np.ones(n)
ub[SiO2_mask.flatten()] = 0.0

if(CHECKMAS):
    sample_optimizedbeta = []
try:
    for _ in range(num_betas):
        print("Starting %i run of adjoint optimization, time is: %f"%(_  ,time.time()), flush=True)
        solver = nlopt.opt(nlopt.LD_MMA, n)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
        solver.set_maxeval(update_factor)
        x[:] = solver.optimize(x)
        if(CHECKMAS):
            sample_optimizedbeta.append(x)
        cur_beta = cur_beta * beta_scale
except KeyboardInterrupt:
    if(CHECKMAS):
        pbar.close()
    
if(CHECKMAS):
    pbar.close()

obj = f(x, np.zeros(n), cur_beta)
print("Step 4 done: optimization loop completed.")
print("Final objective:", obj, flush=True)

if(CHECKMAS):
    np.savez(
        next_opt_filename(),
        Parameter = SaveParameters,
        Message = mes,
        Geo = x,
        Allgeo =sample_optimizedbeta,
        Objective = obj,
        Beta = cur_beta,
        eta_i = eta_i,
        loss = evalarr,
        )
