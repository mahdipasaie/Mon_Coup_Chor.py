import dolfin as df
import copy
import fenics as fe
from dolfin import (
    NonlinearProblem, UserExpression, MeshFunction, FunctionSpace, Function, MixedElement,
    TestFunctions, TrialFunction, split, derivative, NonlinearVariationalProblem,
    NonlinearVariationalSolver, cells, grad, project, refine, Point, RectangleMesh,
    as_vector, XDMFFile, DOLFIN_EPS, sqrt, conditional, Constant, inner, Dx, lt,
    set_log_level, LogLevel, MPI, UserExpression, LagrangeInterpolator, SubDomain
)
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from Mon_Coupl  import *
from ModAdMon import refine_mesh
from mpi4py import MPI


set_log_level(LogLevel.ERROR)

#################### Define Parallel Variables ####################
# Get the global communicator
comm = MPI.COMM_WORLD

# Get the rank of the process
rank = comm.Get_rank()

# Get the size of the communicator (total number of processes)
size = comm.Get_size()
#############################  END  #########


def refine_mesh_local( mesh , rad , center , Max_level  ): 

    xc , yc = center

    mesh_itr = mesh

    for i in range(Max_level):

        mf = MeshFunction("bool", mesh_itr, mesh_itr.topology().dim() , False )

        cells_mesh = cells( mesh_itr )


        index = 0 

        for cell in cells_mesh :

            if ( cell.midpoint()[0] - xc ) **2  + ( cell.midpoint()[1] - yc ) **2  <   1.2 * rad**2 : 





                mf.array()[ index ] = True

            index = index + 1 


        mesh_r = refine( mesh_itr, mf )

        # Update for next loop
        mesh_itr = mesh_r


    return mesh_itr 

physical_parameters_dict = {
    "dy": 0.6 ,
    "max_level": 2,
    "Nx": 100,
    "Ny": 100,
    "dt": 5E-2,
    "dy_coarse":lambda max_level, dy: 2**max_level * dy,
    "Domain": lambda Nx, Ny: [(0.0, 0.0), (Nx, Ny)],
    "a1": 0.8839,
    "a2": 0.6637,
    "w0": 1,
    "tau_0": 1,
    "d0": lambda w0: w0 / 10,
    "W0_scale": 25.888e-8,
    "tau_0_scale": 1.6381419166815996e-6,
    "ep_4": 0.05,
    "k_eq": 0.14,
    "lamda": lambda a1: a1 * 10,
    "D": lambda a2, lamda: a2 * lamda,
    "at": 1 / (2 * fe.sqrt(2.0)),
    "opk": lambda k_eq: 1 + k_eq,
    "omk": lambda k_eq: 1 - k_eq,
    "c_initial": 4,
    "u_initial": -0.2,
    "initial_seed_radius": 8.2663782447466,
    "seed_center": [50, 70],
    "rel_tol": 1E-4,
    "abs_tol": 1E-5,
    "gravity": lambda tau_0_scale, W0_scale: -10 * (tau_0_scale**2) / (W0_scale ) ,
    "rho1": 2.45,
    "rho2": 2.7,
    "mu_fluid": lambda tau_0_scale, W0_scale: 5e-7 * (tau_0_scale) / (W0_scale ** 2),
    "alpha_c": 9.2e-3,
    "viscosity_solid": lambda mu_fluid: mu_fluid *100 ,
    "viscosity_liquid": lambda mu_fluid: mu_fluid,
    "lid_vel_x": 0.0, 
    "lid_vel_y": -0.1,
}


# Extracting parameters from the dictionary
dy = physical_parameters_dict["dy"]
dt = physical_parameters_dict["dt"]
a1 = physical_parameters_dict["a1"]
a2 = physical_parameters_dict["a2"]
w0 = physical_parameters_dict["w0"]
tau_0 = physical_parameters_dict["tau_0"]
W0_scale = physical_parameters_dict["W0_scale"]
tau_0_scale = physical_parameters_dict["tau_0_scale"]
ep_4 = physical_parameters_dict["ep_4"]
k_eq = physical_parameters_dict["k_eq"]
at = physical_parameters_dict["at"]
c_initial = physical_parameters_dict["c_initial"]
u_initial = physical_parameters_dict["u_initial"]
initial_seed_radius = physical_parameters_dict["initial_seed_radius"]
seed_center = physical_parameters_dict["seed_center"]
rel_tol = physical_parameters_dict["rel_tol"]
abs_tol = physical_parameters_dict["abs_tol"]
rho1 = physical_parameters_dict["rho1"]
rho2 = physical_parameters_dict["rho2"]
alpha_c = physical_parameters_dict["alpha_c"]
max_level= physical_parameters_dict["max_level"]
Nx = physical_parameters_dict["Nx"]
Ny= physical_parameters_dict["Ny"]

# Calculated values
d0 = physical_parameters_dict["d0"](w0)
lamda = physical_parameters_dict["lamda"](a1)
D = physical_parameters_dict["D"](a2, lamda)
opk = physical_parameters_dict["opk"](k_eq)
omk = physical_parameters_dict["omk"](k_eq)
gravity = physical_parameters_dict["gravity"](tau_0_scale, W0_scale)
mu_fluid = physical_parameters_dict["mu_fluid"](tau_0_scale, W0_scale)
viscosity_solid = physical_parameters_dict["viscosity_solid"](mu_fluid)
viscosity_liquid = physical_parameters_dict["viscosity_liquid"](mu_fluid)
dy_coarse = physical_parameters_dict["dy_coarse"](max_level, dy)
Domain = physical_parameters_dict["Domain"](Nx, Ny)




nx = (int)(Nx/ dy ) 
ny = (int)(Ny / dy ) 

nx_coarse = (int)(Nx/ dy_coarse ) 
ny_coarse = (int)(Ny / dy_coarse ) 


coarse_mesh = fe.RectangleMesh( df.Point(0.0 , 0.0 ), df.Point(Nx, Ny), nx_coarse, ny_coarse  )
mesh = refine_mesh_local( coarse_mesh , initial_seed_radius , seed_center , max_level  )

# mesh = fe.RectangleMesh( df.Point(0.0 , 0.0 ), df.Point(Nx, Ny), nx, ny  )
# mesh = refine_mesh_local( mesh , initial_seed_radius , seed_center , 1  )



################################## Writing to file #############################



file_N = fe.XDMFFile("Mon__Settel.xdmf" ) 

variable_names = [ "phi", "U", "vel", "press" ] 


def write_simulation_data(Sol_Func, time, file, variable_names, viscosity ):


    
    # Configure file parameters
    file.parameters["rewrite_function_mesh"] = True
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True

    # Split the combined function into its components
    functions = Sol_Func.split(deepcopy=True)

    # Check if the number of variable names matches the number of functions
    if variable_names and len(variable_names) != len(functions):
        raise ValueError("The number of variable names must match the number of functions.")

    # Rename and write each function to the file
    for i, func in enumerate(functions):
        name = variable_names[i] if variable_names else f"Variable_{i}"
        func.rename(name, "solution")
        file.write(func, time)

    func.rename("viscosity", "viscosity")

    file.write(viscosity, time)

    file.close()



phi_u_v_p_Old= None
phi_u_v_p_0_Old= None
T = 0

for it in tqdm(range(0, 10000000)):


    T = T + dt




    # Solving Navier - Stockes Equations
    parameters_dict = update_solver( Domain, mesh, physical_parameters_dict, phi_u_v_p_0_Old )
    solver = parameters_dict["solver"]
    phi_u_v_p = parameters_dict["phi_u_v_p"]
    phi_u_v_p_0 = parameters_dict["phi_u_v_p_0"]
    viscosity_func = parameters_dict["viscosity_func"]
    spaces = parameters_dict["spaces"]


    
    # solving
    solver.solve()
    phi_u_v_p_0.vector()[:] = phi_u_v_p.vector()  # update the solution

    # phi_u_v_p_0_Old = Function(phi_u_v_p_0.function_space())
    # phi_u_v_p_0_Old.assign(phi_u_v_p_0)

    phi_u_v_p_0_Old= phi_u_v_p_0.copy(deepcopy=True)



    if it% 5 == 0 : 

        write_simulation_data( phi_u_v_p_0, T, file_N , variable_names,  viscosity_func)


    # Adaptive Refinement:
    # if it == 5 or it % 30 == 25 :
    #     mesh, mesh_info = refine_mesh(coarse_mesh,  phi_u_v_p_0, spaces, max_level, comm)

    #     if rank == 0 : 
    #         for key, value in mesh_info.items():
    #             print(f"{key}: {value}", flush= True)

    



    














