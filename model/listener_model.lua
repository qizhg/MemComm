
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

function g_build_listener_model()
	 --input table
    local localmap = nn.Identity()() --(#batch, num_channels, visibility*2+1, visibility*2+1)
    local symbol = nn.Identity()() --(#batch, num_symbols) 1-hot vectors
    g_listener_modules['symbol'] = symbol.data.module
    local prev_hid = nn.Identity()() --(#batch, lstm_hidsz)
    g_listener_modules['prev_hid'] = prev_hid.data.module
    local prev_cell = nn.Identity()() --(#batch, lstm_hidsz)
    g_listener_modules['prev_cell'] = prev_cell.data.module

    --game parameters
    local num_channels = 3 + g_opts.num_types_objects * 2
    local visibility = g_opts.listener_visibility
    local num_symbols = g_opts.num_symbols

    --apply conv-fc to localmap
    local n_featuremaps = {3, 16, 32}
    local filter_size =   {1, 1, 1}
    local filter_stride = {1, 1, 1}

    local conv1 = nn.SpatialConvolution(num_channels, n_featuremaps[1], 
                                filter_size[1], filter_size[1], 
                                filter_stride[1], filter_stride[1])(localmap)
    local nonl1 = nonlin()(conv1)
    
    local conv2 = nn.SpatialConvolution(n_featuremaps[1], n_featuremaps[2], 
                                filter_size[2], filter_size[2], 
                                filter_stride[2], filter_stride[2])(nonl1)
    local nonl2 = nonlin()(conv2)
    
    local conv3 = nn.SpatialConvolution(n_featuremaps[2], n_featuremaps[3], 
                                filter_size[3], filter_size[3], 
                                filter_stride[3], filter_stride[3])(nonl2)
    local nonl3 = nonlin()(conv3)

    local out_dim = 1 * 1 * n_featuremaps[3]
    local fc_view = nn.View(out_dim):setNumInputDims(3)(nonl3)
    local localmap_embedding = nonlin()(nn.Linear(out_dim, g_opts.hidsz)(fc_view))

    --apply embedding to 1-hot symbol
    local symbol_embedding = nn.Linear(num_symbols, g_opts.hidsz)(symbol)

    --concat 2 embeddings as input to lstm
    local lstm_input = nn.JoinTable(2)({localmap_embedding,symbol_embedding}) --(#batch, g_opts.hidsz*2)
    local lstm_hidsz = g_opts.hidsz*2
    local hidstate, cellstate = build_lstm(lstm_input, prev_hid, prev_cell, lstm_hidsz)

    --apply fc to lstm hid to get action_prob and baseline
    local hid_act = nonlin()(nn.Linear(lstm_hidsz, lstm_hidsz)(hidstate))
    local action = nn.Linear(lstm_hidsz, g_opts.listener_nactions)(hid_act)
    local action_prob = nn.LogSoftMax()(action)
    local hid_bl = nonlin()(nn.Linear(lstm_hidsz, lstm_hidsz)(hidstate))
    local baseline = nn.Linear(lstm_hidsz, 1)(hid_bl)
    
    local model = nn.gModule({localmap, symbol, prev_hid, prev_cell},
                             {action_prob, baseline, hidstate, cellstate})
    return model
end


function g_init_listener_model()
    g_listener_modules = {}
    g_listener_model = g_build_listener_model()
    g_listener_paramx, g_listener_paramdx = g_listener_model:getParameters()
    if g_opts.init_std > 0 then
        g_listener_paramx:normal(0, g_opts.init_std)
    end
    g_listener_bl_loss = nn.MSECriterion()
end