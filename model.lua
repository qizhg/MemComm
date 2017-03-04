
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
    --input (g_opts.memsize, in_dim)
    --context (1, hidsz)
    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords


    local A_Linear = nn.LookupTable(in_dim, g_opts.hidsz)(input) --(memsize, hidsz)
    share('A_Linear', A_Linear)
    g_modules['A_Linear'] = A_Linear

    local B_Linear = nn.LookupTable(in_dim, g_opts.hidsz)(input) --(memsize, hidsz)
    share('B_Linear', B_Linear)
    g_modules['B_Linear'] = B_Linear

    local MMaout = nn.MM(false, true)
    local Aout = MMaout({context, A_Linear}) --(1, memsize)
    local P = nn.SoftMax()(Aout) --(1, memsize)
    g_modules[hop]['prob'] = P.data.module
    local MMbout = nn.MM(false, false)
    return MMbout({P, B_Linear})
end


local function build_memory(input, context)
    local hid = {}
    hid[0] = context

    for h = 1, g_opts.nhop do
        g_modules[h] = {}
        local Bout = build_lookup_bow(hid[h-1], input, h)
        local C = nn.Linear(g_opts.hidsz, g_opts.hidsz)(hid[h-1])
        share('proj', C)
        local D = nn.CAddTable()({C, Bout})
        hid[h] = nonlin()(D)
    end
    return hid
end

local function build_model_memnn()

    local input = nn.Identity()() --(g_opts.memsize, in_dim)
    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords

    --context: linear transform of the last observation
    local last_obs = nn.Narrow(1, -1, 1)(input) --(1, in_dim)
    local context  = nn.Linear(in_dim, g_opts.hidsz)(last_obs)

    local hid = build_memory(input, context)

    return input, hid[#hid]
end

function g_build_model()

    local input, output
    g_shareList = {}
    g_modules = {}

    input, output = build_model_memnn()

    local hid_act = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(output))
    local action = nn.Linear(g_opts.hidsz, g_opts.nactions)(hid_act)
    local action_logprob = nn.LogSoftMax()(action)
    local hid_bl = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(output))
    local baseline = nn.Linear(g_opts.hidsz, 1)(hid_bl)

    local model = nn.gModule({input}, {action_logprob, baseline})

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

