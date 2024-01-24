import fenics as fe
import dolfin as df
import time
import numpy as np
from dolfin import LagrangeInterpolator, refine, MeshFunction, split, grad, MPI
from fenics import project, MixedElement, FunctionSpace, TestFunctions, Function, derivative, Constant
from fenics import NonlinearVariationalProblem, NonlinearVariationalSolver


def Value_Coor_dof(phi_u_v_p, v_project_phi, comm):
    """Return value of the solution at the degrees of freedom and corresponding coordinates."""
    V = v_project_phi
    phi_answer, u_answer, v_next, v_tent, p_answer = split(phi_u_v_p)
    coordinates_of_all = V.tabulate_dof_coordinates()
    grad_Phi = project(fe.sqrt(fe.dot(grad(phi_answer), grad(phi_answer))), V)
    phi_value_on_dof = grad_Phi.vector().get_local()

    all_Val_dof = comm.gather(phi_value_on_dof, root=0)
    all_point = comm.gather(coordinates_of_all, root=0)

    # Broadcast the data to all processors
    all_point = comm.bcast(all_point, root=0)
    all_Val_dof = comm.bcast(all_Val_dof, root=0)

    # Combine the data from all processors
    all_Val_dof_1 = [val for sublist in all_Val_dof for val in sublist]
    all_point_1 = [point for sublist in all_point for point in sublist]

    point = np.array(all_point_1)
    Val_dof = np.array(all_Val_dof_1)

    return Val_dof, point


def Coordinates_Of_Int(phi, V_project, comm):
    """Get the small mesh and return coordinates of the interface."""
    dof_Val, dof_Coor = Value_Coor_dof(phi, V_project, comm)
    Index_list = np.concatenate(np.argwhere(dof_Val > 0.01))
    Coord_L_Of_Int = dof_Coor[Index_list]

    return Coord_L_Of_Int


def mark_coarse_mesh(mesh_coarse, Coor):
    """Mark the cells in the coarse mesh that have the interface points in them so they can be refined."""
    mf = MeshFunction("bool", mesh_coarse, mesh_coarse.topology().dim(), False)
    len_mf = len(mf)
    Cell_Id_List = []

    tree = mesh_coarse.bounding_box_tree()

    for Cr in Coor:
        cell_id = tree.compute_first_entity_collision(df.Point(Cr))
        if cell_id != 4294967295 and 0 <= cell_id < len_mf:
            Cell_Id_List.append(cell_id)

    Cell_Id_List = np.unique(np.array(Cell_Id_List, dtype=int))
    mf.array()[Cell_Id_List] = True

    return mf


def refine_to_min(mesh_coarse, phi_u_v_p, V_project, comm):
    """Refine coarse mesh cells that contain the interface coordinate."""
    Coord_L_Of_Int = Coordinates_Of_Int(phi_u_v_p, V_project, comm)
    mf = mark_coarse_mesh(mesh_coarse, Coord_L_Of_Int)
    mesh_new = refine(mesh_coarse, mf)

    return mesh_new



def refine_mesh(coarse_mesh ,  phi_u_v_p_0, spaces, max_level, comm ):


    start = time.perf_counter()

    # Make deep copies of the current and previous solutions
    phi_u_v_p_0_Old= phi_u_v_p_0.copy(deepcopy=True)
    v_project_phi = spaces[0]

    coarse_mesh_it = coarse_mesh

    # Refine the mesh up to the maximum level specified
    for res in range(max_level):
        mesh_new = refine_to_min(coarse_mesh_it, phi_u_v_p_0_Old ,v_project_phi, comm)
        coarse_mesh_it = mesh_new

    # Collect mesh information
    n_cells = df.MPI.sum(comm, mesh_new.num_cells())
    hmin = df.MPI.min(comm, mesh_new.hmin())
    hmax = df.MPI.max(comm, mesh_new.hmax())
    dx_min = hmin / df.sqrt(2)
    dx_max = hmax / df.sqrt(2)

    mesh_info = {
    "n_cells": n_cells,
    "dx_min": dx_min,
    "dx_max": dx_max
    }

    # Collect mesh information
    n_cells = df.MPI.sum(comm, mesh_new.num_cells())
    hmin = df.MPI.min(comm, mesh_new.hmin())
    hmax = df.MPI.max(comm, mesh_new.hmax())
    dx_min = hmin / df.sqrt(2)
    dx_max = hmax / df.sqrt(2)



    return mesh_new, mesh_info
