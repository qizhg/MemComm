require'optim'
g_disp = require('display')

function train_batch(task_id)
	local batch = batch_init(g_opts.batch_size, task_id)
	
    -- record the episode
    local active = {}
    local reward = {}


    local speaker = {}
    local speaker_hidsz = g_opts.hidsz
    speaker.map = {}
    speaker.hid = {} 
    speaker.cell = {}
    speaker.hid[0] = torch.Tensor(#batch, speaker_hidsz):fill(0)
    speaker.cell[0] = torch.Tensor(#batch, speaker_hidsz):fill(0)

    local listener = {}
    local listener_hidsz = 2*g_opts.hidsz
    listener.localmap  = {}
    listener.symbol    = {}
    listener.hid = {} 
    listener.cell = {}
    listener.hid[0] = torch.Tensor(#batch, listener_hidsz):fill(0)
    listener.cell[0] = torch.Tensor(#batch, listener_hidsz):fill(0)
    listener.action = {}


    local Gumbel_noise = {}

    --play the game (forward pass)
    for t = 1, g_opts.max_steps do
        g_disp.image(batch[1].map:to_image())
        active[t] = batch_active(batch)

        --speaker
    	speaker.map[t] = batch_speaker_map(batch,active[t])
        Gumbel_noise[t] = torch.rand(#batch, g_opts.num_symbols):log():neg():log():neg()
        local speaker_out = g_speaker_model[task_id]:forward({speaker.map[t], speaker.hid[t-1],speaker.cell[t-1], Gumbel_noise[t]})
        ----speaker_out  = {Gumbel_SoftMax, hidstate, cellstate}
        speaker.hid[t] = speaker_out[2]:clone()
        speaker.cell[t] = speaker_out[3]:clone()

        --speaker-listener comm
        listener.symbol[t] = speaker_out[1]:clone()
        --listener forward and act
        listener.localmap[t] = batch_listener_localmap(batch,active[t])
        local listener_out = g_listener_model:forward({listener.localmap[t], listener.symbol[t],  listener.hid[t-1],listener.cell[t-1]})
        ----  listener_out = {action_prob, baseline, hidstate, cellstate}
        listener.hid[t] = listener_out[3]:clone()
        listener.cell[t] = listener_out[4]:clone()
        listener.action[t] = sample_multinomial(torch.exp(listener_out[1]))  --(#batch, 1)

        batch_listener_act(batch, listener.action[t], active[t])
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t])
    end
    local success = batch_success(batch)

    --backward pass
    g_listener_paramdx:zero()
    g_speaker_paramdx[task_id]:zero()
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
        local listener_baseline = listener_out[2] --(#batch, 1)
        local R = reward_sum:clone() --(#batch, )
        listener_baseline:cmul(active[t]) --(#batch, 1) 
        R:cmul(active[t]) --(#batch, )
        local listener_grad_baseline = g_listener_bl_loss:backward(listener_baseline, R):mul(g_opts.alpha) --(#batch, 1)
        
        --  using REINFORCE
        local listener_grad_action = torch.Tensor(#batch, g_opts.listener_nactions):zero()
        local R_action = listener_baseline - R
        listener_grad_action:scatter(2, listener.action[t], R_action)
        listener_grad_action:div(#batch)

        --listener_backward and cache grad_hid, grad_cell, grad symbol
        g_listener_model:backward({listener.localmap[t], listener.symbol[t],  listener.hid[t-1],listener.cell[t-1]},
                                {listener_grad_action, listener_grad_baseline, listener_grad_hid, listener_grad_cell})
        
        listener_grad_hid = g_listener_modules['prev_hid'].gradInput:clone()
        listener_grad_cell = g_listener_modules['prev_cell'].gradInput:clone()
        grad_symbol = g_listener_modules['symbol'].gradInput:clone()

        --speaker
        local speaker_out = g_speaker_model[task_id]:forward({speaker.map[t], speaker.hid[t-1],speaker.cell[t-1], Gumbel_noise[t]})
        ----speaker_out  = {Gumbel_SoftMax, hidstate, cellstate}
        g_speaker_model[task_id]:backward({speaker.map[t], speaker.hid[t-1],speaker.cell[t-1], Gumbel_noise[t]},
                                {grad_symbol, speaker_grad_hid, speaker_grad_cell})
        speaker_grad_hid = g_speaker_modules[task_id]['prev_hid'].gradInput:clone()
        speaker_grad_cell = g_speaker_modules[task_id]['prev_cell'].gradInput:clone()
    end

    local stat={}
    stat.reward = reward_sum:sum()
    stat.success = success:sum()
    stat.count = g_opts.batch_size
    return stat
end

function train(N)
	local threashold = 20
    for n = 1, N do
        local stat = {} --for the epoch
		for k = 1, g_opts.nbatches do
            local task_id = 1 --all game in a batch share the same task
            local s = train_batch(task_id)
            g_update_speaker_param(g_speaker_paramx[task_id], g_speaker_paramdx[task_id], task_id)
            g_update_listener_param(g_listener_paramx, g_listener_paramdx)

            for k, v in pairs(s) do
                stat[k] = (stat[k] or 0) + v
            end
        end
        for k, v in pairs(stat) do
            if string.sub(k, 1, 5) == 'count' then
                local s = string.sub(k, 6)
                stat['reward' .. s] = stat['reward' .. s] / v
                stat['success' .. s] = stat['success' .. s] / v
            end
        end
        if stat.success > threashold/100.0 then
            g_opts.save = 'model_at'..threashold
            g_save_model()
            threashold  = threashold + 10
        end

        stat.epoch = #g_log + 1
        print(format_stat(stat))
        table.insert(g_log, stat)
    end

end

function g_update_speaker_param(x, dx, task_id)
   
    local f = function(x0) return x, dx end
    if not g_optim_speaker_state then
        g_optim_speaker_state = {}
        for i = 1, g_opts.num_tasks do
            g_optim_speaker_state[i] = {} 
        end
    end
    local config = {learningRate = g_opts.lrate}
    if g_opts.optim == 'sgd' then
        config.momentum = g_opts.momentum
        config.weightDecay = g_opts.wdecay
        optim.sgd(f, x, config, g_optim_speaker_state[task_id])
    elseif g_opts.optim == 'rmsprop' then
        config.alpha = g_opts.rmsprop_alpha
        config.epsilon = g_opts.rmsprob_eps
        config.weightDecay = g_opts.wdecay
        optim.rmsprop(f, x, config, g_optim_speaker_state[task_id])
    elseif g_opts.optim == 'adam' then
        config.beta1 = g_opts.adam_beta1
        config.beta2 = g_opts.adam_beta2
        config.epsilon = g_opts.adam_eps
        optim.adam(f, x, config, g_optim_speaker_state[task_id])
    else
        error('wrong optim')
    end

end

function g_update_listener_param(x, dx)
   
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
