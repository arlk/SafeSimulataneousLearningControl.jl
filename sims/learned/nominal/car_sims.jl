import Random: seed!

seed!(424)

scatter([x0[1]], [x0[2]], markercolor=:black, markersize=3)

for (i,z) in enumerate(eachrow(traj))
    xc = z[1]
    yc = z[2]
    r = sqrt(exp(-2*0.5*tspan[2]*(i-1)/(N-1))*0.5 + ζ1(85)) + 0.01
    plot!(t->xc+r*cos(t), t->yc+r*sin(t), 0, 2pi, leg=false, linecolor=:wheat, fill=(0, :wheat))
end

scatter!([x0[1]], [x0[2]], markercolor=:black, markersize=3)
scatter!([xf[1]], [xf[2]], markercolor=:black, markersize=3)

for i = 1:length(obs.radius)
    xc = obs.x[i]
    yc = obs.y[i]
    plot!(t->xc+0.9*cos(t), t->yc+0.9*sin(t), 0, 2pi, leg=false, fill=(0, Gray(0.7)), line=(0.5, :black))
end

plot!(traj[:,1], traj[:,2], line=(2, :black, :dashdot))

for i = 1:1
    xact = x0 + 0.5*normalize(randn(n))*(rand()^(1/n))
    l1_sol = solve(l1_sys, xact, tspan, Rodas4(), progress = true, progress_steps = 1)
    plot!(l1_sol, vars=[(1,2)], line=(0.75, :royalblue))
end
plot!(aspect_ratio=:equal, xlim = (-1, 13), ylim = (-3.5, 2.5))
savefig("dubins_rl1.pdf")
plot!()
