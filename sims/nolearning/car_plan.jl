using TrajectoryOptimization
using RobotDynamics
using Altro
using StaticArrays
using LinearAlgebra
using Plots
# Planning

struct ModDubinsCar{F} <: AbstractModel
    hμ::F
end

ModDubinsCar() = ModDubinsCar((x)->0.0)

function RobotDynamics.dynamics(model::ModDubinsCar, x, u)
    @SVector[x[4]*cos(x[3]), x[4]*sin(x[3]), u[1] + model.hμ(x), u[2]]
end

n,m,N = 4,2,101
tspan = (0., 8.)
RobotDynamics.state_dim(::ModDubinsCar) = n
RobotDynamics.control_dim(::ModDubinsCar) = m
Q = Diagonal(@SVector fill(0.1,n))
R = Diagonal(@SVector fill(0.1,m))
xf = @SVector [12.,-1.,0.,1.5]
obj = LQRObjective(Q, 10*R, N*Q, xf, N)

# Create our list of constraints
cons = ConstraintList(n,m,N)
goalcon = GoalConstraint(xf)
add_constraint!(cons, goalcon, N)  # add to the last time step

#  Create control limits
xmin = SA[-Inf, -Inf, -pi, 1]
xmax = SA[ Inf,  Inf,  pi, 2]
umin = SA[-1, -0.5]
umax = SA[ 1,  0.5]
u_bnd = BoundConstraint(n,m, x_min = xmin, x_max = xmax, u_min = umin, u_max = umax)
add_constraint!(cons, u_bnd, 1:N-1)  # add to all but the last time step

# Obstacles
obs = CircleConstraint(n, SA[4.,7.,9.], SA[-1.,0.,-2], 1.0*SA[1., 1., 1.])
add_constraint!(cons, obs, 2:N-1)

# Create problem
model = ModDubinsCar()
# Intial condition of the system
x0 = [0., 0., 0., 1.2]
prob = Problem(model, obj, xf, tspan[2], constraints=cons, x0=x0)
solver = ALTROSolver(prob)
solve!(solver)         # solve with ALTRO

# Get the state and control trajectories
X = states(solver)
U = controls(solver)
traj = hcat(X...)'
plot(traj[:,1], traj[:,2], line=(1, :black, :dashdot))

for i = 1:length(obs.radius)
    xc = obs.x[i]
    yc = obs.y[i]
    r = obs.radius[i]
    plot!(t->xc+r*cos(t), t->yc+r*sin(t), 0, 2pi, leg=false, fill=(0, Gray(0.7)), line=(0.5, :black))
end
plot!(aspect_ratio=:equal)


