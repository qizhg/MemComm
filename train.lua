require 'optim'
require('nn')
require('nngraph')
--g_disp = require('display')

local function build_Gumbel_SoftMax(logp, noise)
    local noise = noise or nn.Identity()()
    local logp = logp or nn.Identity()()
    local Gumbel_trick = nn.CAddTable()({noise, logp})
    local Gumbel_trick_temp = nn.MulConstant(1.0/g_opts.Gumbel_temp)(Gumbel_trick)
    local Gumbel_SoftMax = nn.SoftMax()(Gumbel_trick_temp)
    local model = nn.gModule({logp, noise}, {Gumbel_SoftMax})

    return model
end

function train_batch(task_id)
	local batch = batch_init(g_opts.batch_size, task_id)
    local num_batchs = (epoch_num-1)*g_opts.nbatches + batch_num

	
    -- record the episode
    local active = {}
    local reward = {}
    local baseline = {}


    local speaker = {}
    local speaker_hidsz = g_opts.hidsz
    speaker.map = {}
    speaker.hid = {} 
    speaker.cell = {}
    speaker.hid[0] = torch.Tensor(#batch, speaker_hidsz):fill(0)
    speaker.cell[0] = torch.Tensor(#batch, speaker_hidsz):fill(0)

    local listener = {}
    local listener_hidsz = 2 * g_opts.hidsz
    listener.localmap  = {}
    listener.symbol    = {}
    listener.hid = {} 
    listener.cell = {}
    listener.hid[0] = torch.Tensor(#batch, listener_hidsz):fill(0)
    listener.cell[0] = torch.Tensor(#batch, listener_hidsz):fill(0)
    listener.action = {}


    local Gumbel_noise = {}
    local symbol_logp = {}

    --play the game (forward pass)
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)

        --speaker
    	speaker.map[t] = batch_speaker_map(batch,active[t])
        local speaker_out = g_speaker_model:forward(speaker.map[t])
        ----speaker_out  = action_prob
        --comm
        listener.symbol[t] = speaker_out:clone()
        --listener forward and act
        listener.localmap[t] = batch_listener_localmap(batch, active[t])
        local listener_out = g_listener_model:forward({listener.localmap[t], listener.symbol[t],  listener.hid[t-1],listener.cell[t-1]})
        ----listener_out = {action_prob, baseline, hidstate, cellstate}
        baseline[t] = listener_out[2]:clone():cmul(active[t])
        listener.hid[t] = listener_out[3]:clone()
        listener.cell[t] = listener_out[4]:clone()
        listener.action[t] = sample_multinomial(torch.exp(listener_out[1]))  --(#batch, 1)

        batch_listener_act(batch, listener.action[t], active[t])
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t])
    end
    local success = batch_success(batch)

    --prepare for GAE
    local delta = {} --TD residual
    delta[g_opts.max_steps] = reward[g_opts.max_steps] - baseline[g_opts.max_steps]
    for t=1, g_opts.max_steps-1 do 
        delta[t] = reward[t] + g_opts.gamma*baseline[t+1] - baseline[t]
    end
    local A_GAE={} --GAE advatage
    A_GAE[g_opts.max_steps] = delta[g_opts.max_steps]
    for t=g_opts.max_steps-1, 1, -1 do 
        A_GAE[t] = delta[t] + g_opts.gamma*g_opts.lambda*A_GAE[t+1] 
    end

    --backward pass
    g_listener_paramdx:zero()
    g_speaker_paramdx:zero()
    local listener_grad_hid = torch.Tensor(#batch, listener_hidsz):fill(0)
    local listener_grad_cell = torch.Tensor(#batch, listener_hidsz):fill(0)
    local speaker_grad_hid = torch.Tensor(#batch, speaker_hidsz):fill(0)
    local speaker_grad_cell = torch.Tensor(#batch, speaker_hidsz):fill(0)

    local reward_sum = torch.Tensor(#batch):zero() --running reward sum

    for t = g_opts.max_steps, 1, -1 do
        reward_sum:add(reward[t])

        --listener
        local listener_out = g_listener_model:forward({listener.localmap[t], listener.symbol[t],  listener.hid[t-1],listener.cell[t-1]})
        ----  listener_out = {action_logprob, baseline, hidstate, cellstate}
        
        ---- compute listener_grad baseline
        local R = reward_sum:clone() --(#batch, )
        R:cmul(active[t]) --(#batch, )
        local listener_grad_baseline = g_listener_bl_loss:backward(baseline[t], R):mul(g_opts.alpha):div(#batch) --(#batch, 1)
        
        ----  compute listener_grad_action via GAE
        local listener_grad_action = torch.Tensor(#batch, g_opts.listener_nactions):zero()
        listener_grad_action:scatter(2, listener.action[t], A_GAE[t]:view(-1,1):neg())
        ----  compute listener_grad_action with entropy regularization
        local beta = g_opts.beta_start - num_batchs*g_opts.beta_start/g_opts.beta_end_batch
        beta = math.max(0,beta)
        local logp = listener_out[1]
        local entropy_grad = logp:clone():add(1)
        entropy_grad:cmul(torch.exp(logp))
        entropy_grad:mul(beta)
        entropy_grad:cmul(active[t]:view(-1,1):expandAs(entropy_grad):clone())
        listener_grad_action:add(entropy_grad)
        listener_grad_action:div(#batch)

        --listener_backward and cache grad_hid, grad_cell, grad symbol
        g_listener_model:backward({listener.localmap[t], listener.symbol[t],  listener.hid[t-1],listener.cell[t-1]},
                                {listener_grad_action, listener_grad_baseline, listener_grad_hid, listener_grad_cell})
        
        listener_grad_hid = g_listener_modules['prev_hid'].gradInput:clone()
        listener_grad_cell = g_listener_modules['prev_cell'].gradInput:clone()
        local grad_symbol_logp = g_listener_modules['symbol'].gradInput:clone()

        --comm

        --speaker
        local speaker_out = g_speaker_model:forward(speaker.map[t])
        ----speaker_out  = symbol_logprob
        g_speaker_model:backward(speaker.map[t],grad_symbol_logp)
    end

    local stat={}
    stat.reward = reward_sum:sum()
    stat.success = success:sum()
    stat.count = g_opts.batch_size
    return stat
end

function train_batch_thread(opts_orig, listener_paramx_orig, speaker_paramx_orig,task_id)
    g_opts = opts_orig
    g_listener_paramx:copy(listener_paramx_orig)
    g_speaker_paramx:copy(speaker_paramx_orig)
    local stat = train_batch(task_id)
    return g_listener_paramdx, g_speaker_paramdx, stat
end

-- EVERYTHING ABOVE RUNS ON THREADS

function train(N)
	local threashold = 20
    for n = 1, N do
        epoch_num = n
        local stat = {} --for the epoch
		for k = 1, g_opts.nbatches do
            xlua.progress(k, g_opts.nbatches)
            batch_num = k
            local task_id = 1 --all game in a batch share the same task
            if g_opts.nworker > 1 then
                g_listener_paramdx:zero()
                g_speaker_paramdx:zero()
                for w = 1, g_opts.nworker do
                    g_workers:addjob(w, train_batch_thread,
                        function(listener_paramdx_thread, speaker_paramdx_thread, s)
                            g_listener_paramdx:add(listener_paramdx_thread)
                            g_speaker_paramdx:add(speaker_paramdx_thread)
                            merge_stat(stat, s)
                        end, g_opts, g_listener_paramx, g_speaker_paramx, task_id)
                end
                g_workers:synchronize()
            else
                local s = train_batch(task_id)
                merge_stat(stat, s)
            end
            g_update_speaker_param(g_speaker_paramx, g_speaker_paramdx, task_id)
            g_update_listener_param(g_listener_paramx, g_listener_paramdx)
        end
        
        for k, v in pairs(stat) do
            if string.sub(k, 1, 5) == 'count' then
                local s = string.sub(k, 6)
                stat['reward' .. s] = stat['reward' .. s] / v
                stat['success' .. s] = stat['success' .. s] / v
            end
        end

        if stat.bl_count ~= nil and stat.bl_count > 0 then
            stat.bl_cost = stat.bl_cost / stat.bl_count
        else
            stat.bl_cost = 0
        end

        if stat.success > threashold/100.0 then
            g_opts.save = 'model_comm_at'..threashold
            g_save_model()
            threashold  = threashold + 10
        end

        stat.epoch = #g_log + 1
        print(format_stat(stat))
        table.insert(g_log, stat)
    end

end

function g_update_speaker_param(x, dx, task_id)
    dx:div(g_opts.nworker)
    local f = function(x0) return x, dx end
    if not g_optim_speaker_state then
        g_optim_speaker_state = {}
        for i = 1, g_opts.num_tasks do
            g_optim_speaker_state[i] = {} 
        end
    end
    
    local num_batchs = (epoch_num-1)*g_opts.nbatches + batch_num
    local lr = g_opts.lrate_start - num_batchs*g_opts.lrate_start/g_opts.lrate_end_batch
    lr = math.max(0,lr)
    local config = {learningRate = g_opts.lrate}
    
    if g_opts.optim == 'sgd' then
        config.momentum = g_opts.momentum
        config.weightDecay = g_opts.wdecay
        optim.sgd(f, x, config, g_optim_speaker_state)
    elseif g_opts.optim == 'rmsprop' then
        config.alpha = g_opts.rmsprop_alpha
        config.epsilon = g_opts.rmsprob_eps
        config.weightDecay = g_opts.wdecay
        optim.rmsprop(f, x, config, g_optim_speaker_state)
    elseif g_opts.optim == 'adam' then
        config.beta1 = g_opts.adam_beta1
        config.beta2 = g_opts.adam_beta2
        config.epsilon = g_opts.adam_eps
        optim.adam(f, x, config, g_optim_speaker_state)
    else
        error('wrong optim')
    end

end

function g_update_listener_param(x, dx)
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
