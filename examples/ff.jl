using Flux, Pickle
using LinearAlgebra
using BSON: @save

function torchparams(pkl)
    torch_1_w = pkl["0.weight"]
    torch_1_b = pkl["0.bias"]
    torch_2_w = pkl["2.weight"]
    torch_params = params(torch_1_w, torch_1_b, torch_2_w)
end

pkl = Pickle.Torch.THload("data/car.pth.tar")
nn_W = Chain(Dense(2,128,tanh),Dense(128,16))
Flux.loadparams!(nn_W, torchparams(pkl["model_W"]))
nn_Wbot = Chain(Dense(1,128,tanh),Dense(128,4))
Flux.loadparams!(nn_Wbot, torchparams(pkl["model_Wbot"]))

@save "data/car_W.bson" nn_W
@save "data/car_Wbot.bson" nn_Wbot
