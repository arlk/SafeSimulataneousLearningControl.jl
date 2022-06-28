import Random: seed!

seed!(424)

scatter([x0[1]], [x0[2]], markercolor=:black, markersize=3)
scatter!([xf[1]], [xf[2]], markercolor=:black, markersize=3)

for i = 1:length(obs.radius)
    xc = obs.x[i]
    yc = obs.y[i]
    r = 0.9*obs.radius[i]
    plot!(t->xc+r*cos(t), t->yc+r*sin(t), 0, 2pi, leg=false, fill=(0, Gray(0.7)), line=(0.5, :black))
end

for i = 1:10
    xact = x0 + 0.5*normalize(randn(n))*(rand()^(1/n))
    ptb_sol = solve(ptb_sys, xact, tspan, Tsit5(), progress = true, progress_steps = 1)
    # check if colliding
    collision = 0
    for (i,xt) in enumerate(ptb_sol.u), j = 1:length(obs.radius)
        xc = obs.x[j]
        yc = obs.y[j]
        r = 0.9*obs.radius[j]
        if (xt[1] - xc)^2 + (xt[2] - yc)^2 < r^2
            collision = i
            break
        end
    end

    if collision == 0
        plot!(ptb_sol, vars=[(1,2)], line=(0.75, :royalblue))
    else
        xt = [x[1] for x in ptb_sol.u[1:collision]]
        yt = [x[2] for x in ptb_sol.u[1:collision]]
        plot!(xt, yt, line=(0.75, :royalblue))
        scatter!([xt[end]], [yt[end]], markershape=:diamond, markercolor=:red, markersize=3)
    end
end
plot!(traj[:,1], traj[:,2], line=(2, :black, :dashdot), label='x')

plot!(aspect_ratio=:equal, xlim = (-1, 13), ylim = (-3.5, 2.5))
savefig("dubins_ccm.pdf")
