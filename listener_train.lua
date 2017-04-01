require 'optim'
require('nn')
require('nngraph')
--g_disp = require('display')

function listener_train_batch(task_id)
	local batch = batch_init(g_opts.batch_size, task_id)
	
    -- record the episode
    local active = {}
    local symbol = {}
    

    local listener = {}
    local listener_hidsz = 2 * g_opts.hidsz
    listener.localmap  = {}
    listener.symbol_onehot    = {}
    listener.hid = {} 
    listener.cell = {}
    listener.hid[0] = torch.Tensor(#batch, listener_hidsz):fill(0)
    listener.cell[0] = torch.Tensor(#batch, listener_hidsz):fill(0)
    listener.action = {}

    --play the game (forward pass)
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)
        symbol[t] = torch.LongTensor(#batch,1):random(1,g_opts.num_symbols)
        listener.symbol_onehot[t] = torch.Tensor(#batch, g_opts.num_symbols):zero()
        local ones = torch.ones(#batch,1)
        listener.symbol_onehot[t]:scatter(2, symbol[t], ones)

        --listener forward and act
        listener.localmap[t] = batch_listener_localmap(batch,active[t])
        local listener_out = g_listener_model:forward({listener.localmap[t], listener.symbol_onehot[t],  listener.hid[t-1],listener.cell[t-1]})
        ----listener_out = {action_prob, baseline, hidstate, cellstate}
        listener.hid[t] = listener_out[3]:clone()
        listener.cell[t] = listener_out[4]:clone()
        listener.action[t] = sample_multinomial(torch.exp(listener_out[1]))  --(#batch, 1)
        batch_listener_act(batch, listener.action[t], active[t])
        batch_update(batch, active[t])
    end

    --backward pass
    g_listener_paramdx:zero()
    local listener_grad_hid = torch.Tensor(#batch, listener_hidsz):fill(0)
    local listener_grad_cell = torch.Tensor(#batch, listener_hidsz):fill(0)
    local avg_err = 0
    for t = g_opts.max_steps, 1, -1 do
        --listener
        local listener_out = g_listener_model:forward({listener.localmap[t], listener.symbol_onehot[t],  listener.hid[t-1],listener.cell[t-1]})
        ----  listener_out = {action_logprob, baseline, hidstate, cellstate}
        
        ---- compute listener_grad baseline
        local listener_grad_baseline = torch.Tensor(#batch, 1):zero()

        ----  compute listener_grad_action via NLL
        local NLLcriterion = nn.ClassNLLCriterion()
        local err = NLLcriterion:forward(listener_out[1], symbol[t]:squeeze())
        avg_err = avg_err + err
        local listener_grad_action = NLLcriterion:backward(listener_out[1], symbol[t]:squeeze())

        --listener_backward and cache grad_hid, grad_cell, grad symbol
        g_listener_model:backward({listener.localmap[t], listener.symbol_onehot[t],  listener.hid[t-1],listener.cell[t-1]},
                                {listener_grad_action, listener_grad_baseline, listener_grad_hid, listener_grad_cell})
        
        listener_grad_hid = g_listener_modules['prev_hid'].gradInput:clone()
        listener_grad_cell = g_listener_modules['prev_cell'].gradInput:clone()
        local grad_symbol = g_listener_modules['symbol'].gradInput:clone()
    end

    local stat={}
    stat.reward = -avg_err/g_opts.max_steps
    return stat
end

function listener_train(N)
    for n = 1, N do
        local stat = {} --for the epoch
		for k = 1, g_opts.nbatches do
            local task_id = 1 --all game in a batch share the same task
            local s = listener_train_batch(task_id)
            merge_stat(stat, s)
            g_update_listener_param(g_listener_paramx, g_listener_paramdx)

        end
        
        for k, v in pairs(stat) do
            if string.sub(k, 1, 5) == 'count' then
                local s = string.sub(k, 6)
                stat['reward' .. s] = stat['reward' .. s] / v
                stat['success' .. s] = stat['success' .. s] / v
            end
        end

        stat.epoch = #g_log + 1
        print(format_stat(stat))
        table.insert(g_log, stat)
    end

end
local function update_listener_param(x, dx)
    dx:div(g_opts.nworker)   
    local f = function(x0) return x, dx end
    if not g_optim_listener_state then g_optim_listener_state = {} end
    local config = {learningRate = g_opts.lrate}
    if g_opts.optim == 'sgd' then
        config.momentum = g_opts.momentum
        config.weightDecay = g_opts.wdecay
        optim.sgd(f, x, config, g_optim_listener_state)
    elseif g_opts.optim == 'rmsprop' then
        config.alpha = g_opts.rmsprop_alpha
        config.epsilon = g_opts.rmsprob_eps
        config.weightDecay = g_opts.wdecay
        optim.rmsprop(f, x, config, g_optim_listener_state)
    elseif g_opts.optim == 'adam' then
        config.beta1 = g_opts.adam_beta1
        config.beta2 = g_opts.adam_beta2
        config.epsilon = g_opts.adam_eps
        optim.adam(f, x, config, g_optim_listener_state)
    else
        error('wrong optim')
    end

end
