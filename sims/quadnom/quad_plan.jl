using TrajectoryOptimization
using RobotDynamics
using Altro
using StaticArrays
using LinearAlgebra
using Plots
# Planning

struct ModQuadrotor{F} <: AbstractModel
    hμ::F
end

ModQuadrotor() = ModQuadrotor((x)->[0.0, 0.0, 0.0])

function RobotDynamics.dynamics(model::ModQuadrotor, x, u)
    val = model.hμ(x)
    #  dx4 = -x[7]*sin(x[9])
    #  dx5 = x[7]*cos(x[9])*sin(x[8])
    #  dx6 = 9.81 - x[7]*cos(x[9])*cos(x[8])
    #  @SVector[x[4], x[5], x[6], dx4, dx5, dx6, val[1] + u[1], val[2] + u[2], val[3] + u[3]]
    @SVector[x[4], x[5], x[6], 9.81*tan(x[7]), 9.81*tan(x[8]), val[1] + u[1], val[2] + u[2], val[3] + u[3]]
end

n,m,N = 8,3,101
tspan = (0., 6.)
RobotDynamics.state_dim(::ModQuadrotor) = n
RobotDynamics.control_dim(::ModQuadrotor) = m
Q = Diagonal(@SVector fill(1e-1,n))
R = Diagonal(@SVector fill(1e-2,m))
Qf = Diagonal(@SVector fill(1e2,n))
xf = @SVector [0.,5.,1.,0.,0.,0.,0.,0.]
obj = LQRObjective(Q, R, Qf, xf, N)

# Create our list of constraints
cons = ConstraintList(n,m,N)
goalcon = GoalConstraint(xf)
add_constraint!(cons, goalcon, N)  # add to the last time step

#  Create control limits
xmin = SA[-Inf, -Inf, -Inf, -2, -2, -2, -pi/3, -pi/3]
xmax = SA[ Inf,  Inf,  Inf,  2,  2,  2,  pi/3,  pi/3]
umin = SA[-5, -pi/3, -pi/3]
umax = SA[ 5,  pi/3,  pi/3]
u_bnd = BoundConstraint(n,m, x_min = xmin, x_max = xmax, u_min = umin, u_max = umax)
add_constraint!(cons, u_bnd, 1:N-1)  # add to all but the last time step

# Create problem
model = ModQuadrotor()
# Intial condition of the system
x0 = [0.0,-5.,1.,0.,0.,0.,0.,0.]
prob = Problem(model, obj, xf, tspan[2], constraints=cons, x0=x0)

# Maze {{{
function maze()
    r_quad_maze = 1.0
    r_cylinder_maze = 0.5
    maze_cylinders = []
    for i = range(-0.5,stop=3,length=24) # middle obstacle
        push!(maze_cylinders,(i, 0, r_cylinder_maze))
    end
    for i = range(-4,stop=-2.5,length=12) # middle obstacle
        push!(maze_cylinders,(i, 0, r_cylinder_maze))
    end
    #  push!(maze_cylinders,(0, 0, r_cylinder_maze))

    n_maze_cylinders = length(maze_cylinders)
    maze_xyr = collect(zip(maze_cylinders...))

    cx = SVector{n_maze_cylinders}(maze_xyr[1])
    cy = SVector{n_maze_cylinders}(maze_xyr[2])
    cr = SVector{n_maze_cylinders}(maze_xyr[3])

    CircleConstraint(n, cx, cy, cr .+ r_quad_maze)
end
# }}}

obs = maze()
add_constraint!(cons, obs, 2:N-1)

# phantom
obs2 = CircleConstraint(n, SA[-2.0, -1.5, -1.], SA[0., 0., 0.], SA[2.5, 2.5, 2.5])
add_constraint!(cons, obs2, 2:N-1)

X_guess = zeros(n,4)
X_guess[:,1] = x0
X_guess[:,4] = xf
wpts = [6 -2 1;
        6  2 1]
X_guess[1:3,2:3] .= wpts'
#
X0 = Altro.interp_rows(N,tspan[2],X_guess)
initial_states!(prob, X0)

solver = ALTROSolver(prob)

solve!(solver)         # solve with ALTRO

# Get the state and control trajectories
X = states(solver)
U = controls(solver)
traj = hcat(X...)'

plot()
for (i,z) in enumerate(eachrow(traj))
    xc = z[1]
    yc = z[2]
    r = 1.0 # sqrt(exp(-2*0.5*tspan[2]*(i-1)/(N-1))*0.5 + ζ1(80)) + 0.01
    plot!(t->xc+r*cos(t), t->yc+r*sin(t), 0, 2pi, leg=false, linecolor=:wheat, fill=(0, :wheat))
end

for i = 1:length(obs.radius)
    xc = obs.x[i]
    yc = obs.y[i]
    r = 0.5
    plot!(t->xc+r*cos(t), t->yc+r*sin(t), 0, 2pi, leg=false, fill=(0, Gray(0.7)), line=(0.0, Gray(0.7)))
end
plot!(traj[:,1], traj[:,2], line=(1, :black, :dashdot))
plot!(aspect_ratio=:equal)


