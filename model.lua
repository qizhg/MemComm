
require('nn')
require('nngraph')

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

local function build_encoder(hidsz)
	-- linear encoder
	local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords
	local m = nn.Linear(in_dim, hidsz)
	g_modules['encoder_linear'] = m --nn module to encode raw input
	return m
end


local function build_lookup_bow(context, input, hop)
   --input: (#batch, memsize, in_dim)
   --context (#batch, hidsz)
    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords

    local A_in_table = {}
    local B_in_table = {}
    for i =1, g_opts.batch_size do
        local input2dim = nn.Select(1,i)(input) --(memsize, in_dim)
        
        local A_Linear = nn.Linear(in_dim, g_opts.hidsz)(input2dim)
        table.insert(A_in_table, nn.View(1, g_opts.memsize, g_opts.hidsz)(A_Linear) )--(1, memsize, hidsz)
        share('A_Linear', A_Linear)
        g_modules['A_Linear'] = A_Linear
         

        local B_Linear = nn.Linear(in_dim, g_opts.hidsz)(input2dim)
        table.insert(B_in_table, nn.View(1, g_opts.memsize, g_opts.hidsz)(B_Linear) )--(1, memsize, hidsz)
        share('B_Linear', B_Linear)
        g_modules['B_Linear'] = B_Linear
    end
    local Ain = nn.JoinTable(1)(A_in_table) --(#batch, memsize, in_dim)
    local Bin = nn.JoinTable(1)(B_in_table) --(#batch, memsize, in_dim)
    

    
    local context3dim = nn.View(1, -1):setNumInputDims(1)(context) --(#batch, 1, hidsz)
    local Aout = nn.MM(false, true)({context3dim, Ain}) --(#batch, 1, memsize)
    local P = nn.SoftMax()(Aout) --(#batch, 1, memsize)
    g_modules[hop]['prob'] = P.data.module
    local Bout3dim = nn.MM(false, false)({P, Bin}) --(#batch, 1, hidsz)

    return nn.Squeeze()(Bout3dim)  --(#batch, hidsz)
end


local function build_memory(input, context)
    --input: (#batch, memsize, in_dim)
    --context: (#batch, hidsz)
    local hid = {}
    hid[0] = context --(#batch, hidsz)

    for h = 1, g_opts.nhop do
        g_modules[h] = {}
        local Bout = build_lookup_bow(hid[h-1], input, h) --(#batch, hidsz)
        local C = nn.Linear(g_opts.hidsz, g_opts.hidsz)(hid[h-1]) --(#batch, hidsz)
        share('proj', C)
        local D = nn.CAddTable()({C, Bout}) --(#batch, hidsz)
        hid[h] = nonlin()(D) --(#batch, hidsz)
    end
    return hid
end

local function build_model_memnn()

    local input = nn.Identity()()  --(#batch, memsize, in_dim)
    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords

    local last_obs = nn.Identity()() --(#batch, in_dim)

    --context: linear transform of the last observation
    local context  = nn.Linear(in_dim, g_opts.hidsz)(last_obs) --(#batch, hidsz)

    local hid = build_memory(input, context)

    return {input, last_obs}, hid[#hid]
end

function g_build_model()

    local input, output
    g_shareList = {}
    g_modules = {}

    input_table, output = build_model_memnn()

    --input_table = {input, last_obs}
    ----input (#batch, memsize, in_dim)
    ----last_obs (#batch, in_dim)
    local hid_act = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(output)) --(#batch, hidsz)
    local action = nn.Linear(g_opts.hidsz, g_opts.nactions)(hid_act) --(#batch, nactions)
    local action_logprob = nn.LogSoftMax()(action)  --(#batch, nactions)
    local hid_bl = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(output))  --(#batch, hidsz)
    local baseline = nn.Linear(g_opts.hidsz, 1)(hid_bl) --(#batch, 1)

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
end

