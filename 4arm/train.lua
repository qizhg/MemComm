

require'optim'


function train_batch()
	---get a new game
	local batch = batch_init(g_opts.batch_size)


	-- record the episode
    local active = {}
    local reward = {}
    local action = {}


    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords
    local obs = torch.Tensor(g_opts.max_steps, #batch, in_dim)
    obs:fill(0)


    --play the game (forward pass)
    local game = batch[1]
    local agent = game.agent
    --print('agent start: '..agent.loc.y..'  '..agent.loc.x)
    --print('des: '..agent.route[#agent.route].y..'  '..agent.route[#agent.route].x)
    
    for t = 1, g_opts.max_steps do

        --get active games
        active[t] = batch_active(batch)
    	--get the latest observation
    	obs[t] = batch_obs(batch,active[t])
        --get input
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


        

        --forward input to get output = {action_logprob, baseline}
        local out
        if g_opts.memsize >0 then 
            out = g_model:forward({mem_input, new_obs})  --out[1] = action_logprob
        else
            out = g_model:forward(new_obs)
        end

        ep = g_opts.eps_start - (g_nbatches-1)*(g_opts.eps_start-g_opts.eps_end)/(g_opts.eps_end_batch-1)
        --ep = math.max(ep, g_opts.eps_end)
        if torch.uniform() < ep then
            action[t] = torch.LongTensor(#batch,1)
            action[t]:random(1, g_opts.nactions)
        else
            action[t] = sample_multinomial(torch.exp(out[1]))  --(#batch, 1)
        end

        
        --if t==1 and agent.route[#agent.route].y==11 and agent.route[#agent.route].x==6 then
        --    print(torch.exp(out[1][1]))
        --end

        
       
        batch_act(batch, action[t], active[t])
        --print('agent at: '..agent.loc.y..'  '..agent.loc.x)
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t]) --(#batch, )

        --print(reward[t][1])
    end
    --print('agent end: '..agent.loc.y..'  '..agent.loc.x)
    --local temp = io.read("*n")
    local success = batch_success(batch)
    --print(success[1])

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

end

function train(N)

    local to_update= true
    local threashold = 70
	
    for n = 1, N do
        local stat = {} --for the epoch
		for k = 1, g_opts.nbatches do
            g_nbatches = (n-1)*g_opts.nbatches + k
			local s = train_batch() --get g_paramx, g_paramdx
            if to_update then
                g_update_param(g_paramx, g_paramdx)
            end
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
            --g_n_70 = g_n_70+1
            g_opts.save = 'mem'..g_opts.memsize..'at'..threashold
            g_save_model()
            threashold  = threashold + 10
        end

        stat.epoch = #g_log + 1
        print(format_stat(stat))
        print(ep)
        table.insert(g_log, stat)
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
