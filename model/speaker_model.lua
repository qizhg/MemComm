
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

local function build_lstm(input, prev_hid, prev_cell, hidsz)
    local pre_hid = {}
    table.insert(pre_hid, nn.Linear(hidsz, hidsz * 4)(input))
    table.insert(pre_hid, nn.Linear(hidsz, hidsz * 4)(prev_hid))
    local preactivations = nn.CAddTable()(pre_hid)
    -- gates
    local pre_sigmoid_chunk = nn.Narrow(2, 1, 3 * hidsz)(preactivations)
    local all_gates = nn.Sigmoid()(pre_sigmoid_chunk)

    -- input
    local in_chunk = nn.Narrow(2, 3 * hidsz + 1, hidsz)(preactivations)
    local in_transform = nn.Tanh()(in_chunk)

    local in_gate = nn.Narrow(2, 1, hidsz)(all_gates)
    local forget_gate = nn.Narrow(2, hidsz + 1, hidsz)(all_gates)
    local out_gate = nn.Narrow(2, 2 * hidsz + 1, hidsz)(all_gates)

    -- previous cell state contribution
    local c_forget = nn.CMulTable()({forget_gate, prev_cell})
    -- input contribution
    local c_input = nn.CMulTable()({in_gate, in_transform})
    -- next cell state
    local cellstate = nn.CAddTable()({
      c_forget,
      c_input
    })
    local c_transform = nn.Tanh()(cellstate)
    local hidstate = nn.CMulTable()({out_gate, c_transform})
        
    return hidstate, cellstate
end

function g_build_speaker_model()
	--input table
    local map = nn.Identity()() --(#batch, num_channels, map_height, map_width)

    --game parameters
    local num_channels
    if g_opts.pickup_enable == true then
        num_channels = 3 + g_opts.num_types_objects * 2 --3: block, water, listener
    else
        num_channels = 3 + g_opts.num_types_objects
    end

    --apply conv-fc to map
    g_opts.n_featuremaps = {3, 16, 64}
    g_opts.filter_size =   {1, 1, 3}
    g_opts.filter_stride = {1, 1, 1}
    local n_featuremaps = g_opts.n_featuremaps
    local filter_size = g_opts.filter_size
    local filter_stride = g_opts.filter_stride
    local d = g_opts.map_height

    local conv1 = nn.SpatialConvolution(num_channels, n_featuremaps[1], 
                                filter_size[1], filter_size[1], 
                                filter_stride[1], filter_stride[1])(map)
    local nonl1 = nonlin()(conv1)
    
    local conv2 = nn.SpatialConvolution(n_featuremaps[1], n_featuremaps[2], 
                                filter_size[2], filter_size[2], 
                                filter_stride[2], filter_stride[2])(nonl1)
    local nonl2 = nonlin()(conv2)

    local pool2 = nn.SpatialMaxPooling(2, 2, 2, 2)(nonl2)
    d = math.floor(d / 2)
    
    local conv3 = nn.SpatialConvolution(n_featuremaps[2], n_featuremaps[3], 
                                filter_size[3], filter_size[3], 
                                filter_stride[3], filter_stride[3])(pool2)
    local nonl3 = nonlin()(conv3)
    d = d - 2 

    local out_dim = d * d * n_featuremaps[3]
    local fc_view = nn.View(out_dim):setNumInputDims(3)(nonl3)
    local map_embedding = nonlin()(nn.Linear(out_dim, g_opts.hidsz)(fc_view))

    local hid_act = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(map_embedding))
    local action = nn.Linear(g_opts.hidsz, g_opts.num_symbols)(hid_act)
    local action_prob = nn.LogSoftMax()(action)
    local hid_bl = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(map_embedding))
    local baseline = nn.Linear(g_opts.hidsz, 1)(hid_bl)
    local model = nn.gModule({map}, {action_prob, baseline})
    return model
end

function g_build_speaker_model()
    --input table
    local map = nn.Identity()() --(#batch, num_channels, map_height, map_width)

    --game parameters
    local num_channels
    if g_opts.pickup_enable == true then
        num_channels = 3 + g_opts.num_types_objects * 2 --3: block, water, listener
    else
        num_channels = 3 + g_opts.num_types_objects
    end

    --apply conv-fc to map
    local n_featuremaps = {3, 16, 64}
    local filter_size =   {1, 1, 3}
    local filter_stride = {1, 1, 1}
    local d = g_opts.map_height

    local conv1 = nn.SpatialConvolution(num_channels, n_featuremaps[1], 
                                filter_size[1], filter_size[1], 
                                filter_stride[1], filter_stride[1])(map)
    local nonl1 = nonlin()(conv1)
    
    local conv2 = nn.SpatialConvolution(n_featuremaps[1], n_featuremaps[2], 
                                filter_size[2], filter_size[2], 
                                filter_stride[2], filter_stride[2])(nonl1)
    local nonl2 = nonlin()(conv2)

    --local pool2 = nn.SpatialMaxPooling(2, 2, 2, 2)(nonl2)
    --d = math.floor(d / 2)
    
    local conv3 = nn.SpatialConvolution(n_featuremaps[2], n_featuremaps[3], 
                                filter_size[3], filter_size[3], 
                                filter_stride[3], filter_stride[3])(nonl2)
    local nonl3 = nonlin()(conv3)
    d = d - 2 

    local out_dim = d * d * n_featuremaps[3]
    local fc_view = nn.View(out_dim):setNumInputDims(3)(nonl3)
    local map_embedding = nonlin()(nn.Linear(out_dim, g_opts.hidsz)(fc_view))

    if g_opts.lstm == false then 
        local hid_act = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(map_embedding))
        local action = nn.Linear(g_opts.hidsz, g_opts.num_symbols)(hid_act)
        local action_prob = nn.LogSoftMax()(action)
        local hid_bl = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(map_embedding))
        local baseline = nn.Linear(g_opts.hidsz, 1)(hid_bl)
        local model = nn.gModule({map}, {action_prob, baseline})
        return model
    else --lstm with hidsz
        local prev_hid = nn.Identity()() --(#batch, lstm_hidsz)
        g_speaker_modules['prev_hid'] = prev_hid.data.module
        local prev_cell = nn.Identity()() --(#batch, lstm_hidsz)
        g_speaker_modules['prev_cell'] = prev_cell.data.module
        local lstm_input = map_embedding
        local hidstate, cellstate = build_lstm(lstm_input, prev_hid, prev_cell, g_opts.hidsz)

        local hid_act = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(hidstate))
        local action = nn.Linear(g_opts.hidsz, g_opts.num_symbols)(hid_act)
        local action_prob = nn.LogSoftMax()(action)
        local hid_bl = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(hidstate))
        local baseline = nn.Linear(g_opts.hidsz, 1)(hid_bl)
        local model = nn.gModule({map, prev_hid, prev_cell}, {action_prob, baseline, hidstate, cellstate})
        return model
    end
end


function g_init_speaker_model()
    g_speaker_model   = {}
    g_speaker_modules = {}
    g_speaker_paramx  = {}
    g_speaker_paramdx = {}

    g_speaker_model = g_build_speaker_model()
    g_speaker_paramx, g_speaker_paramdx = g_speaker_model:getParameters()
    if g_opts.init_std > 0 then
        g_speaker_paramx:normal(0, g_opts.init_std)
    end
    
    g_speaker_bl_loss = nn.MSECriterion()
end