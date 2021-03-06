
require('nn')
require('nngraph')
paths.dofile('LinearNB.lua')
paths.dofile('Entropy.lua')


local function nonlin()
    if g_opts.nonlin == 'tanh' then
        return nn.Tanh()
    elseif g_opts.nonlin == 'relu' then
        return nn.ReLU()
    elseif g_opts.nonlin == 'none' then
        return nn.Identity()
    else
        error('wrong nonlin')
    end
end

local function share(name, mod)
    if g_shareList[name] == nil then g_shareList[name] = {} end
    table.insert(g_shareList[name], mod)
end


local function build_lookup_bow(context, input, hop)
   --input: (#batch, memsize, in_dim + memsize)
   --context (#batch, hidsz)
    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords + g_opts.memsize

    local A_in_table = {}
    local B_in_table = {}
    for i =1, g_opts.batch_size do
        local input2dim = nn.Select(1,i)(input) --(memsize, in_dim)
        
        local A_Linear = nn.LinearNB(in_dim, g_opts.hidsz)(input2dim)
        table.insert(A_in_table, nn.View(1, g_opts.memsize, g_opts.hidsz)(A_Linear) )--(1, memsize, hidsz)
        share('A_Linear', A_Linear)
        g_modules['A_Linear'] = A_Linear
         

        local B_Linear = nn.LinearNB(in_dim, g_opts.hidsz)(input2dim)
        table.insert(B_in_table, nn.View(1, g_opts.memsize, g_opts.hidsz)(B_Linear) )--(1, memsize, hidsz)
        share('B_Linear', B_Linear)
        g_modules['B_Linear'] = B_Linear
    end
    local Ain = nn.JoinTable(1)(A_in_table) --(#batch, memsize, in_dim)
    local Bin = nn.JoinTable(1)(B_in_table) --(#batch, memsize, in_dim)
    

    
    local context3dim = nn.View(1, -1):setNumInputDims(1)(context) --(#batch, 1, hidsz)
    local Aout = nn.MM(false, true)({context3dim, Ain}) --(#batch, 1, memsize)
    local Aout2dim = nn.View(g_opts.batch_size, memsize)(Aout) --(#batch*1, memsize)
    local P2dim = nn.SoftMax()(Aout2dim) --(#batch*1, memsize)
    local P = nn.View(g_opts.batch_size, 1, g_opts.memsize)(P2dim) --(#batch, 1, memsize)
    g_modules[hop]['prob'] = P.data.module
    local Bout3dim = nn.MM(false, false)({P, Bin}) --(#batch, 1, hidsz)

    return nn.Squeeze()(Bout3dim)  --(#batch, hidsz)
end


local function build_memory(input, context)
    --input: (#batch, memsize, in_dim)
    --context: (#batch, hidsz)
    local hid = {}
    hid[0] = context --(#batch, hidsz)

    local P
    local Bout
    for h = 1, g_opts.nhop do
        g_modules[h] = {}
        Bout = build_lookup_bow(hid[h-1], input, h) --(#batch, hidsz)
        --local Bout = temp[1]
        --P = temp[2]
        
        --local C = nn.LinearNB(g_opts.hidsz, g_opts.hidsz)(hid[h-1]) --(#batch, hidsz)
        --share('proj', C)
        --local D = nn.CAddTable()({C, Bout}) --(#batch, hidsz)
        --hid[h] = nonlin()(D) --(#batch, hidsz)
        hid[h] = Bout
    end
    return hid
end

local function build_model_memnn()

    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords
    local input = nn.Identity()()  --(#batch, memsize, in_dim)
    local new_obs = nn.Identity()() --(#batch, in_dim)

    --context: linear transform of the new observation
    local context  = nn.Linear(in_dim, g_opts.hidsz)(new_obs) --(#batch, hidsz)

    local hid = build_memory(input, context)
    local context_prime = nn.LinearNB(g_opts.hidsz,g_opts.hidsz)(context)
    --local hid = temp[1]
    --local Bout = temp[2]
    local output = nn.CAddTable()({hid[#hid], context_prime})
    return {input, new_obs}, {output}
    
end

local function build_model_non_memnn()

    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords
    local new_obs = nn.Identity()() --(#batch, in_dim)
    local output  = nn.Linear(in_dim, g_opts.hidsz)(new_obs) --(#batch, hidsz)

    return {new_obs}, {output}
end

function g_build_model()

    local input, output
    g_shareList = {}
    g_modules = {}

    local input_table, output_table
    if g_opts.memsize > 0 then
        input_table, output_table = build_model_memnn()
    else
        input_table, output_table = build_model_non_memnn()
    end
    
    output = output_table[1]

    local hid_act = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(output)) --(#batch, hidsz)
    --local  hid_act = output
    local action = nn.Linear(g_opts.hidsz, g_opts.nactions)(hid_act) --(#batch, nactions)
    local action_logprob = nn.LogSoftMax()(action)  --(#batch, nactions)
    local hid_bl = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(output))  --(#batch, hidsz)
    local baseline =  nn.Linear(g_opts.hidsz, 1)(hid_bl) --(#batch, 1)

    local model = nn.gModule(input_table, {action_logprob, baseline})

    for _, l in pairs(g_shareList) do
        if #l > 1 then
            local m1 = l[1].data.module
            for j = 2,#l do
                local m2 = l[j].data.module
                m2:share(m1,'weight','bias','gradWeight','gradBias')
            end
        end
    end

    return model
end

function g_init_model()
    g_model = g_build_model()
    g_paramx, g_paramdx = g_model:getParameters()
    if g_opts.init_std > 0 then
        g_paramx:normal(0, g_opts.init_std)
    end
    g_bl_loss = nn.MSECriterion()
    g_entropy_loss = nn.Entropy()
end

