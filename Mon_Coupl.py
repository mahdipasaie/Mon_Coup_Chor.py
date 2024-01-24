# ############################## Import Needed Dependencies ################################
import dolfin as df
import fenics as fe
from dolfin import (
    NonlinearProblem, UserExpression, MeshFunction, FunctionSpace, Function, MixedElement,
    TestFunctions, TrialFunction, split, derivative, NonlinearVariationalProblem,
    NonlinearVariationalSolver, cells, grad, project, refine, Point, RectangleMesh,
    as_vector, XDMFFile, DOLFIN_EPS, sqrt, conditional, Constant, inner, Dx, lt,
    set_log_level, LogLevel, MPI, UserExpression, LagrangeInterpolator, DirichletBC, near, SubDomain
)

import numpy as np
import matplotlib.pyplot as plt
import time

global_bc = None

class InitialConditions(fe.UserExpression):

    def __init__(self, seed_center, u_initial, initial_circle_radius, **kwargs):

        super().__init__(**kwargs)  # Initialize the base class
        self.rad = initial_circle_radius
        self.u_initial = u_initial
        self.seed_center= seed_center 
        self.xc = seed_center[0]
        self.yc = seed_center[1]


    def eval(self, values, x):
        x_p = x[0]
        y_p = x[1]
        dist = (self.xc-x_p)**2 + (self.yc-y_p)**2
        # PF: 
        values[0] = -np.tanh((fe.sqrt(dist)-self.rad)/fe.sqrt(2.0))
        values[1] = self.u_initial
        # NS : 
        values[2] = 0.0     # x component 
        values[3] = 0.0     # y component
        values[4] = 0.0     # pressure


    def value_shape(self):
        return (5,)


def define_variables(mesh):
    
    P1 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # Order parameter Phi
    P2 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # U: dimensionless solute supersaturation
    P3 = fe.VectorElement("Lagrange", mesh.ufl_cell(), 2)  # Velocity
    P4 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # Pressure 

    viscosity_func_space = FunctionSpace(mesh, P1)
    viscosity_func = Function(viscosity_func_space)

    coeff1_bouyancy_func = Function(viscosity_func_space)
    coeff2_bouyancy_func = Function(viscosity_func_space)


    element = MixedElement( [P1, P2, P3, P4] )

    function_space = FunctionSpace( mesh, element )

    Test_Functions = TestFunctions(function_space)
    v_test_phi, q_test_u, w_test_v, z_test_p = Test_Functions



    phi_u_v_p = Function(function_space)  
    phi_u_v_p_0 = Function(function_space)  

    phi_answer, u_answer, v_answer, p_answer = split(phi_u_v_p)  # Current solution
    phi_prev, u_prev, v_prev, p_prev = split(phi_u_v_p_0)  # Previous solution


    num_subs = function_space.num_sub_spaces()
    spaces, maps = [], []
    for i in range(num_subs):
        space_i, map_i = function_space.sub(i).collapse(collapsed_dofs=True)
        spaces.append(space_i)
        maps.append(map_i)


    return {
        'phi_answer': phi_answer, 'u_answer': u_answer,
        'v_answer': v_answer, 'p_answer' : p_answer, 
        "phi_prev": phi_prev, "u_prev" : u_prev,
        "v_prev": v_prev, "p_prev": p_prev,
        "spaces": spaces, "maps": maps,
        "phi_u_v_p": phi_u_v_p, "phi_u_v_p_0": phi_u_v_p_0, 
        "v_test_phi": v_test_phi, "q_test_u": q_test_u, 
        "w_test_v": w_test_v, "z_test_p": z_test_p, 
        "function_space": function_space, "Test_Functions": Test_Functions, 
         "viscosity_func": viscosity_func, 
          "coeff1_bouyancy_func": coeff1_bouyancy_func, "coeff2_bouyancy_func": coeff2_bouyancy_func
            } 


def set_or_update_initial_conditions( phi_u_v_p_0, seed_center, u_initial ,initial_circle_radius, phi_u_v_p_0_Old ):

    if phi_u_v_p_0_Old is not None:

        LagrangeInterpolator.interpolate(phi_u_v_p_0, phi_u_v_p_0_Old)

    else:

        initial_v = InitialConditions( seed_center ,u_initial, initial_circle_radius, degree=2)
        phi_u_v_p_0.interpolate(initial_v)



    return  phi_u_v_p_0


def update_viscosity( phi_u_v_p_0 , viscosity_func, viscosity_solid, viscosity_liquid ): 
 
    phi_prev, u_prev, v_prev, p_prev = phi_u_v_p_0.split( deepcopy = True )

    phi_values = phi_prev.vector().get_local()
    viscosity_values = viscosity_func.vector().get_local()

    for i in range(len(phi_values)):

        Phi_value =  phi_values[i]
        SP = (1+Phi_value)/2
        SN = (1-Phi_value)/2 
        viscosity_values[i] = viscosity_solid * SP + viscosity_liquid * SN

    # Update the Viscosity function based on the viscosity values assigned previously
    viscosity_func.vector().set_local(viscosity_values)
    viscosity_func.vector().apply('insert')

    return  viscosity_func


def bouyancy_terms_update( phi_u_v_p_0, coeff1_bouyancy_func ,coeff2_bouyancy_func, rho1 , rho2, c_initial, gravity, alpha_c, omk, opk ): 


    phi_prev, u_prev, v_prev, p_prev = phi_u_v_p_0.split( deepcopy = True )

    phi_values = phi_prev.vector().get_local()
    u_values = u_prev.vector().get_local()

    coeff1_values = coeff1_bouyancy_func.vector().get_local()
    coeff2_values = coeff2_bouyancy_func.vector().get_local()

    e_u = u_values * omk + 1 
    term_1 = opk - omk * phi_values
    conctreation = c_initial*e_u/2*term_1

    for i in range(len(phi_values)):

        Phi_value =  phi_values[i]
        SN = (1-Phi_value)/2 
        SP = (1+Phi_value)/2
        coeff1_values[i] = SN* alpha_c* gravity* (conctreation[i]- c_initial ) 
        coeff2_values[i] = SP* (rho2 - rho1 )/ rho1* gravity



    # Update the  function based on the viscosity values assigned previously
    coeff1_bouyancy_func.vector().set_local(coeff1_values)
    coeff1_bouyancy_func.vector().apply('insert')

    coeff2_bouyancy_func.vector().set_local(coeff2_values)
    coeff2_bouyancy_func.vector().apply('insert')


    return  coeff1_bouyancy_func, coeff2_bouyancy_func


def calculate_dependent_variables(phi_answer, W0, ep_4 ):

    # Define tolerance for avoiding division by zero errors
    tolerance_d = fe.sqrt(DOLFIN_EPS)  # sqrt(1e-15)

    # Calculate gradient and derivatives for anisotropy function
    grad_phi = fe.grad(phi_answer)
    mgphi = fe.inner(grad_phi, grad_phi)
    dpx = fe.Dx(phi_answer, 0)
    dpy = fe.Dx(phi_answer, 1)
    dpx = fe.variable(dpx)
    dpy = fe.variable(dpy)

    # Normalized derivatives
    nmx = -dpx / fe.sqrt(mgphi)
    nmy = -dpy / fe.sqrt(mgphi)
    norm_phi_4 = nmx**4 + nmy**4

    # Anisotropy function
    a_n = fe.conditional(
        fe.lt(fe.sqrt(mgphi), fe.sqrt(DOLFIN_EPS)),
        fe.Constant(1 - 3 * ep_4),
        1 - 3 * ep_4 + 4 * ep_4 * norm_phi_4
    )

    # Weight function based on anisotropy
    W_n = W0 * a_n

    # Derivatives of weight function w.r.t x and y
    D_w_n_x = fe.conditional(fe.lt(fe.sqrt(mgphi), tolerance_d), 0, fe.diff(W_n, dpx))
    D_w_n_y = fe.conditional(fe.lt(fe.sqrt(mgphi), tolerance_d), 0, fe.diff(W_n, dpy))

    return { "D_w_n_x": D_w_n_x, "D_w_n_y": D_w_n_y, "mgphi": mgphi,"W_n": W_n }


def epsilon(u ):  

    return 0.5 * (fe.grad(u) + fe.grad(u).T)


def sigma(u, p, mu1, rho1 ):
    # chnged due to kinematic viscosity

    return 2 * mu1 * epsilon(u) - p * (1/rho1)  * fe.Identity(len(u))


def calculate_equation_1(
     phi_answer, phi_prev, u_answer,
    dt, v_test, d_w_n_x, d_w_n_y, mgphi, w_n
    , k_eq, w0, lamda, v_answer, tau_0
):


    term4_in = mgphi * w_n * d_w_n_x
    term5_in = mgphi * w_n * d_w_n_y

    term4 = -fe.inner(term4_in, v_test.dx(0)) * fe.dx
    term5 = -fe.inner(term5_in, v_test.dx(1)) * fe.dx

    term3 = -(w_n**2 * fe.inner(fe.grad(phi_answer), fe.grad(v_test))) * fe.dx

    term2 = (
        fe.inner(
            (phi_answer - phi_answer**3) - lamda * u_answer  * (1 - phi_answer**2) ** 2,
            v_test,
        ) * fe.dx
    )

    tau_n = (w_n / w0) ** 2 * tau_0

    term1 = -fe.inner((tau_n) * (phi_answer - phi_prev) / dt, v_test) * fe.dx

    # Advection term due to velocity of the fluid ( Negetive Cause on the LHS of Eq which goes RHS ) 
    term6 = - fe.inner((tau_n) * fe.dot(v_answer, fe.grad(phi_answer)), v_test) * fe.dx  


    eq1 = term1 + term2 + term3 + term4 + term5 +  term6

    return eq1


def calculate_equation_2(u_answer, u_prev, phi_answer, phi_prev, q_test, dt, D, opk, omk, at, v_answer):

    tolerance_d = fe.sqrt(DOLFIN_EPS)  

    grad_phi = fe.grad(phi_answer)
    abs_grad = fe.sqrt(fe.inner(grad_phi, grad_phi))

    norm = fe.conditional(
        fe.lt(abs_grad, tolerance_d), fe.as_vector([0, 0]), grad_phi / abs_grad
    )

    dphidt = (phi_answer - phi_prev) / dt

    term6 = -fe.inner(((opk) / 2 - (omk) * phi_answer / 2) * (u_answer - u_prev) / dt, q_test) * fe.dx
    term7 = -fe.inner(D * (1 - phi_answer) / 2 * fe.grad(u_answer), fe.grad(q_test)) * fe.dx
    term8 = -at * (1 + (omk) * u_answer) * dphidt * fe.inner(norm, fe.grad(q_test)) * fe.dx
    term9 = (1 + (omk) * u_answer) * dphidt / 2 * q_test * fe.dx

    # Advection Term LHS goes to RHS ( Negative ):  V · {  [ (1 + k - (1 - k)ϕ) / 2]  ∇U -  [(1 + (1 - k)U) / 2]  ∇ϕ } : 

    term10_1 = fe.dot( v_answer ,( opk - omk * phi_answer ) / 2 * fe.grad(u_answer)  ) # V ·  (1 + k - (1 - k)ϕ) / 2]  ∇U

    term10_2 =  fe.dot( v_answer, - (1+ omk *u_answer) / 2 * grad_phi ) # -  V ·  [(1 + (1 - k)U) / 2]  ∇ϕ

    term10 = - fe.inner( term10_1 + term10_2, q_test ) * fe.dx # RHS ( Negative )

    eq2 = term6 + term7 + term8 + term9 + term10

    return eq2


def F1(v_answer, q_test, dt):

    F1 = fe.inner(fe.div(v_answer), q_test) * dt * fe.dx

    return F1


def F2(v_answer, v_prev, p_answer, v_test, dt, rho1, mu1, Coeff1_Bou_NS, Coeff2_Bou_NS ):

    F2 = (
        fe.inner((v_answer - v_prev) / dt, v_test) * fe.dx
        + fe.inner(fe.dot(v_answer, fe.grad(v_answer)), v_test) * fe.dx
        # + (1/rho1) * fe.inner(sigma(u_answer, p_answer, mu1), epsilon(v_test)) * fe.dx
        + fe.inner(sigma(v_answer, p_answer, mu1, rho1), epsilon(v_test)) * fe.dx
        - fe.inner( Coeff1_Bou_NS , v_test[1] ) * fe.dx
        - fe.inner( Coeff2_Bou_NS , v_test[1] ) * fe.dx


    )

    return F2



############################ Boundary Condition Section #################
def Define_Boundary_Condition_NS( function_space, Domain, mesh, lid_vel_x, lid_vel_y  ) :

    global global_bc

    W = function_space

    # Define the Domain boundaries based on the previous setup
    (X0, Y0), (X1, Y1) = Domain

    # Define boundary conditions for velocity, pressure, and temperature
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], X0)

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], X1)

    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Y0)

    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Y1)

    # Instantiate boundary classes
    left_boundary = LeftBoundary()
    right_boundary = RightBoundary()
    bottom_boundary = BottomBoundary()
    top_boundary = TopBoundary()

    # Define Dirichlet boundary conditions
    bc_u_left = DirichletBC(W.sub(2), Constant((0, 0)), left_boundary)
    bc_u_right = DirichletBC(W.sub(2), Constant((0, 0)), right_boundary)
    bc_u_bottom = DirichletBC(W.sub(2), Constant((0, 0)), bottom_boundary)
    bc_u_top = DirichletBC(W.sub(2).sub(1), Constant(lid_vel_y), top_boundary)
    bc_u_top_x = DirichletBC(W.sub(2).sub(0), Constant(lid_vel_x), top_boundary)
    bc_p_bottom = DirichletBC(W.sub(3), Constant(0.0), bottom_boundary)

    zero_pressure_point = fe.Point( (X0),  (Y1) )
    bc_p_zero = DirichletBC(W.sub(3), Constant(0.0), lambda x,
    on_boundary: near(x[0], zero_pressure_point.x()) and near(x[1], zero_pressure_point.y()), method="pointwise")


    # bc_all = [bc_u_left, bc_u_right, bc_u_bottom, bc_u_top, bc_p_zero, bc_u_top_x]
    bc_all = [bc_u_left, bc_u_right, bc_u_top, bc_u_top_x,bc_p_bottom ]


    global_bc = bc_all


    return  bc_all



def total_form_definer( variables_dict, physical_parameters_dict, phi_u_v_p_0_Old  ): 

    # In this function it is assumed that phi_u_v_p_0 is already interpolated with initital condttion 
    # cause the viscosity and bouyancy terms are defined with phi initial value

    phi_u_v_p = variables_dict["phi_u_v_p"]
    phi_u_v_p_0 = variables_dict["phi_u_v_p_0"]
    Test_Functions = variables_dict["Test_Functions"]
    viscosity_func = variables_dict["viscosity_func"]
    function_space = variables_dict["function_space"]
    coeff1_bouyancy_func = variables_dict["coeff1_bouyancy_func"]
    coeff2_bouyancy_func = variables_dict["coeff2_bouyancy_func"]


    phi_answer, u_answer, v_answer, p_answer = split(phi_u_v_p)
    phi_prev, u_prev, v_prev, p_prev = split(phi_u_v_p_0)
    v_test_phi, q_test_u, w_test_v, z_test_p = Test_Functions

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
    lid_vel_x = physical_parameters_dict["lid_vel_x"]
    lid_vel_y = physical_parameters_dict["lid_vel_y"]

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


    # Initial condition:
    phi_u_v_p_0 = set_or_update_initial_conditions( phi_u_v_p_0, seed_center, u_initial ,initial_seed_radius, phi_u_v_p_0_Old )

    # define the viscosity and buoyancy terms:     
    viscosity_func = update_viscosity( phi_u_v_p_0 , viscosity_func, viscosity_solid, viscosity_liquid )
    coeff1_bouyancy_func, coeff2_bouyancy_func = bouyancy_terms_update( phi_u_v_p_0, coeff1_bouyancy_func ,coeff2_bouyancy_func, rho1 , rho2, c_initial, gravity, alpha_c, omk, opk )

    dependent_variables_dict = calculate_dependent_variables(phi_prev, w0, ep_4)

    
    D_w_n_x = dependent_variables_dict["D_w_n_x"]
    D_w_n_y = dependent_variables_dict["D_w_n_y"]
    mgphi = dependent_variables_dict["mgphi"]
    W_n = dependent_variables_dict["W_n"]

    F1_form = calculate_equation_1( phi_answer, phi_prev, u_answer, dt, v_test_phi, D_w_n_x, D_w_n_y, mgphi, W_n, k_eq, w0, lamda, v_answer, tau_0 )
    F2_form = calculate_equation_2( u_answer, u_prev, phi_answer, phi_prev, q_test_u, dt, D, opk, omk, at, v_answer )
    F3_form = F1(v_answer, z_test_p, dt)
    F4_form = F2(v_answer, v_prev, p_prev, w_test_v, dt, rho1, viscosity_func, coeff1_bouyancy_func, coeff2_bouyancy_func )


    total_form = F1_form+ F2_form+ F3_form+ F4_form

    return {"total_form":  total_form, "phi_u_v_p": phi_u_v_p, "phi_u_v_p_0": phi_u_v_p_0, "function_space": function_space, "lid_vel_x": lid_vel_x, "lid_vel_y": lid_vel_y, "rel_tol": rel_tol, "abs_tol": abs_tol, "viscosity_func": viscosity_func  }



def define_problem_solver( phi_u_v_p, total_form, Bc, rel_tol, abs_tol): 
    global global_bc

    J = derivative(total_form, phi_u_v_p)  # Compute the Jacobian
    # Define the problem
    problem = NonlinearVariationalProblem(total_form, phi_u_v_p, global_bc, J=J)
    # Create and configure the solver
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["newton_solver"]["relative_tolerance"] = rel_tol
    prm["newton_solver"]["absolute_tolerance"] = abs_tol
    prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True

    return solver




def update_solver( Domain, mesh, physical_parameters_dict, phi_u_v_p_0_Old):


    variables_dict = define_variables(mesh)
    spaces = variables_dict["spaces"]

    # use total_form_designer and define_problem_solver to update the solver
    dict_form = total_form_definer( variables_dict, physical_parameters_dict, phi_u_v_p_0_Old  )



    total_form =  dict_form["total_form"]
    phi_u_v_p = dict_form["phi_u_v_p"]
    function_space = dict_form["function_space"]
    phi_u_v_p_0 = dict_form["phi_u_v_p_0"]
    rel_tol =  dict_form["rel_tol"]
    abs_tol =  dict_form["abs_tol"]
    viscosity_func = dict_form["viscosity_func"]
    lid_vel_x = dict_form["lid_vel_x"]
    lid_vel_y = dict_form["lid_vel_y"]

    # if phi_u_v_p_0_Old is not None:

    #     LagrangeInterpolator.interpolate(phi_u_v_p_0, phi_u_v_p_0_Old)



    global_bc = Define_Boundary_Condition_NS( function_space, Domain, mesh, lid_vel_x, lid_vel_y )
    solver = define_problem_solver( phi_u_v_p, total_form, global_bc, rel_tol, abs_tol)



    return { "solver": solver, "phi_u_v_p": phi_u_v_p , "phi_u_v_p_0": phi_u_v_p_0, "viscosity_func": viscosity_func, "spaces": spaces }












