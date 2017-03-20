require('nn')
require('nngraph')
paths.dofile('LinearNB.lua')


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
    --context (#batch * nagents, hidsz)
    --input: (#batch * nagents, memsize, in_dim + memsize)
    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords + g_opts.memsize

    local A_in_table = {}
    local B_in_table = {}
    for i =1, g_opts.batch_size*g_opts.nagents do
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
    local Ain = nn.JoinTable(1)(A_in_table) --(#batch* nagents, memsize, in_dim)
    local Bin = nn.JoinTable(1)(B_in_table) --(#batch* nagents, memsize, in_dim)
    

    
    local context3dim = nn.View(1, -1):setNumInputDims(1)(context) --(#batch * nagents, 1, hidsz)
    local Aout = nn.MM(false, true)({context3dim, Ain}) --(#batch* nagents, 1, memsize)
    local Aout2dim = nn.View(g_opts.batch_size*g_opts.nagents, memsize)(Aout) --(#batch * nagents, memsize)
    local P2dim = nn.SoftMax()(Aout2dim) --(#batch*1, memsize)
    local P = nn.View(g_opts.batch_size*g_opts.nagents, 1, g_opts.memsize)(P2dim) --(#batch * nagents, 1, memsize)
    g_modules[hop]['prob'] = P.data.module
    local Bout3dim = nn.MM(false, false)({P, Bin}) --(#batch * nagents, 1, hidsz)

    return nn.Squeeze()(Bout3dim)  --(#batch * nagents, hidsz)
end


local function build_memory(prev_obs, context)
    --prev_obs: (#batch*nagent, memsize, in_dim)
    --context: (#batch * nagents, hidsz)
    local mem_out = {}
    mem_out[0] = context -- (#batch * nagents, hidsz)

    local Bout
    for h = 1, g_opts.nhop do
        g_modules[h] = {}
        Bout = build_lookup_bow(mem_out[h-1], prev_obs, h) --(#batch* nagents, hidsz)
        mem_out[h] = Bout
    end
    return mem_out
end


function g_build_model()
    g_shareList = {}
    g_modules = {}

    --input table
    local prev_obs = nn.Identity()() --(#batch * nagents, memsize, in_dim)
    local cur_obs = nn.Identity()() --(#batch * nagents, in_dim)
    local comm_in = nn.Identity()() -- (#batch * nagents, nagents, hidsz)

    --form context
    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords
    local cur_obs_encoded = nn.Linear(in_dim, g_opts.hidsz)(cur_obs) --(#batch * nagents, hidsz)
    local comm2hid = nn.Sum(2)(comm_in) -- (#batch * nagents, hidsz)
    local context = nonlin() ( nn.CAddTable()({cur_obs_encoded, comm2hid}) ) -- (#batch * nagents, hidsz)

    --query mem

    local mem_out = build_memory(prev_obs, context) --tables of --(#batch * nagents, hidsz)
    local context_prime = nn.LinearNB(g_opts.hidsz,g_opts.hidsz)(context) --(#batch * nagents, hidsz)
    local hid = nonlin() ( nn.CAddTable()({mem_out[#mem_out], context_prime}) ) --(#batch * nagents, hidsz)
    --form action prob & baseline
    local hid2action = nonlin()(nn.Linear(g_opts.hidsz, g_opts.nactions)(hid)) --(#batch * nagents, nactions)
    local action_logprob = nn.LogSoftMax()(hid2action) --(#batch * nagents, nactions)
    local baseline =  nonlin()(nn.Linear(g_opts.hidsz, 1)(hid)) --(#batch  * nagents, 1)

    --form comm_out

    local comm_out = nn.Contiguous()(nn.Replicate(g_opts.nagents, 2)(hid))
    
    local model = nn.gModule({prev_obs, cur_obs, comm_in}, {action_logprob,baseline,comm_out})
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
end