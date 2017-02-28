---rnn + continues comm

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

local function build_encoder(hidsz)
	-- linear encoder
	local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords
	local m = nn.Linear(in_dim, hidsz)
	g_modules['encoder_linear'] = m --nn module to encode raw input
	return m
end

local function build_rnn(input, prev_hid,comm_in)
	-- input is raw, not encoded
	--  hid  <- rnn(input, prev_hid, comm_in)
	-- all are nngraph nodes
	-- input should be encodede first
	local pre_nonlinear = {}
	table.insert(pre_nonlinear, build_encoder(g_opts.hidsz)(input))
	table.insert(pre_nonlinear, nn.Linear(g_opts.hidsz,g_opts.hidsz)(prev_hid))
	g_modules['pre_hid'] = pre_nonlinear[2].data.module --nn module applying to prev_hid
	if comm_in then table.insert(pre_nonlinear, comm_in) end
	local hid = nonlin()(nn.CAddTable()(pre_nonlinear))
	return hid
end

function g_build_model()
	--inputs: input(raw), prev_hid, comm_in
	--outputs: action_prob, baseline, hid, comm_out

	g_model_inputs = {} -- input name(string) -> index
    g_model_outputs = {}
	
	local in_mods = {} --table of input nngraph nodes
	local out_mods = {} --table of output nngraph nodes

	--inputs
	----input(raw), prev_hid
	local prev_hid = nn.Identity()()
	local input = nn.Identity()()
	table.insert(in_mods, input)
	g_model_inputs['input'] = #in_mods
	table.insert(in_mods, prev_hid)
	g_model_inputs['prev_hid'] = #in_mods
	g_modules['prev_hid'] = prev_hid.data.module
	----comm_in
	local comm2hid --???
	if g_opts.comm then
		local comm_in = nn.Identity()()
		table.insert(in_mods, comm_in)
		g_model_inputs['comm_in'] = #in_mods
		g_modules['comm_in'] = comm_in.data.module
		comm2hid = nn.Sum(2)(comm_in) --last dimension size = hidsiz
	end

	--rnn
	local hid = build_rnn(input, prev_hid, comm2hid)

	--action prob output
	local action = nn.Linear(g_opts.hidsz, g_opts.nactions)(hid)
	local action_prob = nn.LogSoftMax()(action)

	--baseline
	local baseline = nn.Linear(g_opts.hidsz, 1)(hid)

	--outputs
	----action_prob, baseline, hid
	table.insert(out_mods, action_prob)
    g_model_outputs['action_prob'] = #out_mods
    table.insert(out_mods, baseline)
    g_model_outputs['baseline'] = #out_mods
    table.insert(out_mods, hid)
    g_model_outputs['hid'] = #out_mods

    ----comm_out
    if g_opts.comm then
        local comm_out
        comm_out = hid

        if g_opts.comm_decoder >= 1 then
        	comm_out =  nn.Linear(g_opts.hidsz, g_opts.hidsz)(comm_out)
            g_modules['comm_decoder'] = comm_out
            if g_opts.comm_decoder == 2 then
               comm_out = nonlin()(comm_out)
            end
        end
        comm_out = nn.Contiguous()(nn.Replicate(g_opts.nagents, 2)(comm_out))

        table.insert(out_mods, comm_out)
        g_model_outputs['comm_out'] = #out_mods
    end

	local m = nn.gModule(in_mods, out_mods)
	return m
end

function g_init_model()
    g_modules = {}
    g_model = g_build_model()
    g_paramx, g_paramdx = g_model:getParameters()
    if g_opts.init_std > 0 then
        g_paramx:normal(0, g_opts.init_std)
    end
    
    g_bl_loss = nn.MSECriterion() --???
    g_bl_loss.sizeAverage = false  --???
end

