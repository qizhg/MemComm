require'optim'


function train_batch(task_id)
	local batch = batch_init(g_opts.batch_size)
	
    -- record the episode
    local active = {}
    local reward = {}


    local speaker = {}
    local speaker_hidsz = g_opts.hidsz
    speaker.map = {}
    speaker.hid = {} 
    speaker.cell = {}
    speaker.hid[0] = torch.zeros(#batch, speaker_hidsz)
    speaker.cell[0] = torch.zeros(#batch, speaker_hidsz)

    local listener = {}
    local listener_hidsz = 2*g_opts.hidsz
    listener.localmap  = {}
    listener.symbol    = {}
    listener.hid = {} 
    listener.cell = {}
    listener.hid[0] = torch.zeros(#batch, listener_hidsz)
    listener.cell[0] = torch.zeros(#batch, listener_hidsz)
    listener.action = {}


    local Gumbel_noise = {}

    --play the game (forward pass)
    for t = 1, g_opts.max_steps do
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
        ----listener_out  = {action_prob, baseline, hidstate, cellstate}
        listener.hid[t] = listener_out[3]:clone()
        listener.cell[t] = listener_out[4]:clone()
        listener.action[t] = sample_multinomial(torch.exp(listener_out[1]))  --(#batch, 1)
        listener.action[t] = torch.squeeze(listener.action[t]) --(#batch, 1)

        batch_listener_act(batch, listener.action[t], active[t])
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t])
    end

    --[[
    --backward pass
    
    g_paramdx:zero()
    local reward_sum = torch.Tensor(#batch):zero() --running reward sum

    for t = g_opts.max_steps, 1, -1 do
        reward_sum:add(reward[t])

        --do step forward
        local mem_input
        if g_opts.memsize > 0 then 
            mem_input = torch.Tensor(g_opts.memsize, #batch, in_dim)
            mem_input:fill(0)
            if g_opts.memsize > 0 and t > 1 then 
                local mem_start, mem_end
                mem_start = math.max(1, t - g_opts.memsize)
                mem_end = math.max(1, t - 1)
                mem_input[{{1,mem_end-mem_start+1}}] = obs[{{mem_start,mem_end}}]
            end
            mem_input = mem_input:transpose(1,2) --(#batch, memsize, in_dim)
            local time_feature = torch.Tensor(#batch, g_opts.memsize, g_opts.memsize)
            for i = 1, #batch do time_feature[i] = torch.eye(g_opts.memsize) end
            mem_input = torch.cat(mem_input,time_feature,3) --(#batch, memsize, in_dim+memsize)
        end
        local new_obs = obs[t]:clone() --(#batch, in_dim)


        if g_opts.memsize >0 then 
            out = g_model:forward({mem_input, new_obs})  --out[1] = action_logprob
        else
            out = g_model:forward(new_obs)
        end

        --compute grad baseline
        local baseline = out[2] --(#batch, 1)
        local R = reward_sum:clone() --(#batch, )
        baseline:cmul(active[t]) --(#batch, 1) 
        R:cmul(active[t]) --(#batch, )
        local grad_baseline = g_bl_loss:backward(baseline, R):mul(g_opts.alpha) --(#batch, 1)

        --REINFORCE
        local grad = torch.Tensor(#batch, g_opts.nactions):zero()
        local R_action = baseline - R
        grad:scatter(2, action[t], R_action)
        grad:div(#batch) --???

        --backward
        g_model:backward({mem_input, new_obs}, {grad, grad_baseline})
    end

    --print(reward_sum[1])

    --log
    local stat={}
    stat.reward = reward_sum:sum()
    stat.success = success:sum()
    stat.count = g_opts.batch_size
    return stat
    --]]

end

function train(N)
	
    for n = 1, N do
		for k = 1, g_opts.nbatches do
            local task_id = torch.random(1, g_opts.num_tasks) --all game in a batch share the same task
			train_batch(task_id)
        end
    end

end

function g_update_param(x, dx)
   
    local f = function(x0) return x, dx end
    if not g_optim_state then g_optim_state = {} end
    local config = {learningRate = g_opts.lrate}
    if g_opts.optim == 'sgd' then
        config.momentum = g_opts.momentum
        config.weightDecay = g_opts.wdecay
        optim.sgd(f, x, config, g_optim_state)
    elseif g_opts.optim == 'rmsprop' then
        config.alpha = g_opts.rmsprop_alpha
        config.epsilon = g_opts.rmsprob_eps
        config.weightDecay = g_opts.wdecay
        optim.rmsprop(f, x, config, g_optim_state)
    elseif g_opts.optim == 'adam' then
        config.beta1 = g_opts.adam_beta1
        config.beta2 = g_opts.adam_beta2
        config.epsilon = g_opts.adam_eps
        optim.adam(f, x, config, g_optim_state)
    else
        error('wrong optim')
    end

end
