require 'optim'
require('nn')
require('nngraph')
--g_disp = require('display')

local function speaker_train_batch(task_id)
	local batch = batch_init(g_opts.batch_size, task_id)
    local num_batchs = (epoch_num-1)*g_opts.nbatches + batch_num


    --for SL

    local SL = false
    local dst = {}
    local a = batch_active(batch)
    local map = batch_speaker_map(batch, a)
    for i = 1, #batch do
        dst[i]= {}
        for y = 1, g_opts.map_height do
            for x = 1, g_opts.map_width do
                if map[i][4][y][x] == 1 then
                    dst[i].y=y
                    dst[i].x=x
                end
            end
        end
    end

	
    -- record the episode
    local active = {}
    local reward = {}
    local action = {}
    local baseline = {}
    local baseline_target = {}


    local speaker = {}
    speaker.loc = {}
    speaker.map = {}
    speaker.hid = {} 
    speaker.cell = {}
    speaker.hid[0] = torch.Tensor(#batch, g_opts.hidsz):fill(0)
    speaker.cell[0] = torch.Tensor(#batch, g_opts.hidsz):fill(0)

    --play the game (forward pass)
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)
    	speaker.map[t] = batch_speaker_map(batch,active[t])
        speaker.loc[t]={}
        for i = 1, #batch do
            speaker.loc[t][i]={}
            speaker.loc[t][i].y = batch[i].listener.loc.y
            speaker.loc[t][i].x = batch[i].listener.loc.x
        end
        local speaker_out, target_out
        if g_opts.lstm == false then
            speaker_out = g_speaker_model:forward(speaker.map[t])
            target_out = g_speaker_model_target:forward(speaker.map[t])
            ----speaker_out  = {symbols_logprob, symbol_baseline}
        else
            speaker_out = g_speaker_model:forward({speaker.map[t], speaker.hid[t-1],speaker.cell[t-1]})
            ----speaker_out  = {symbols_logprob, symbol_baseline, hidstate, cellstate}
            speaker.hid[t] = speaker_out[3]:clone()
            speaker.cell[t] = speaker_out[4]:clone()
        end

        baseline[t] = speaker_out[2]:clone():cmul(active[t])
        baseline_target[t] = target_out[2]:clone():cmul(active[t])


        local ep = g_opts.eps_start - num_batchs*g_opts.eps_start/g_opts.eps_end_batch
        ep = math.max(ep,0.05)
        if torch.uniform() < ep then
            action[t] = torch.LongTensor(#batch,1)
            action[t]:random(1, g_opts.num_symbols)
        else
            action[t] = sample_multinomial(torch.exp(speaker_out[1]))  --(#batch, 1)
        end
        action[t] = sample_multinomial(torch.exp(speaker_out[1])) --(#batch, 1)


        batch_listener_act(batch, action[t], active[t])
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

    --prepar for target
    --local A_target={}
    --A_target[g_opts.max_steps] = reward[g_opts.max_steps] - baseline[g_opts.max_steps]
    --for t=g_opts.max_steps-1, 1, -1 do 
    --    A_target[t] = reward[t] + g_opts.gamma*baseline_target[t+1] - baseline[t]
    --end


    --backward pass
    local stat={}
    local grad_hid = torch.Tensor(#batch, g_opts.hidsz):fill(0)
    local grad_cell = torch.Tensor(#batch, g_opts.hidsz):fill(0)

    g_speaker_paramdx:zero()
    local reward_sum = torch.Tensor(#batch):zero() --running reward sum
    local norm = 0
    local avg_err = 0
    local avg_blcost = 1
    for t = g_opts.max_steps, 1, -1 do
        reward_sum:add(reward[t])

        local speaker_out, target_out
        if g_opts.lstm == false then
            speaker_out =g_speaker_model:forward(speaker.map[t])
        else
            speaker_out = g_speaker_model:forward({speaker.map[t], speaker.hid[t-1],speaker.cell[t-1]})
        end
         ---- compute speaker_grad baseline
        local R = reward_sum:clone() --(#batch, )
        R:cmul(active[t]) --(#batch, )
        stat.bl_cost = (stat.bl_cost or 0) + g_speaker_bl_loss:forward(baseline[t], R)
        stat.bl_count = (stat.bl_count or 0) + active[t]:sum()
        local grad_baseline = g_speaker_bl_loss:backward(baseline[t], R):mul(g_opts.alpha):div(#batch) --(#batch, 1)
    
        ----  compute speaker_grad_action via GAE
        local grad_symbol_logp = torch.Tensor(#batch, g_opts.num_symbols):zero()
        grad_symbol_logp:scatter(2, action[t], A_GAE[t]:view(-1,1):neg())

        ----  compute speaker_grad_action with entropy regularization
        local beta = g_opts.beta_start - num_batchs*g_opts.beta_start/g_opts.beta_end_batch
        beta = math.max(0,beta)
        local logp = speaker_out[1]
        local entropy_grad = logp:clone():add(1)
        entropy_grad:cmul(torch.exp(logp))
        entropy_grad:mul(beta)
        entropy_grad:cmul(active[t]:view(-1,1):expandAs(entropy_grad):clone())
        grad_symbol_logp:add(entropy_grad)
        grad_symbol_logp:div(#batch)

        
        if SL==true then
            --grad_baseline:zero()
            grad_symbol_logp:zero()
            local NLLceriterion = nn.ClassNLLCriterion()
            local action_label = torch.LongTensor(#batch)
            for i = 1, #batch do
                local x = speaker.loc[t][i].x
                local y = speaker.loc[t][i].y
                if y > dst[i].y then 
                    action_label[i] = 1
                elseif y < dst[i].y then 
                    action_label[i] = 2
                elseif x > dst[i].x then 
                    action_label[i] = 3
                elseif x < dst[i].x then 
                    action_label[i] = 4
                else 
                    action_label[i] = 1
                end
            end

            local err = NLLceriterion:forward(speaker_out[1],action_label)
            avg_err = avg_err +err
            grad_symbol_logp = NLLceriterion:backward(speaker_out[1],action_label)
            grad_symbol_logp:div(#batch)
        end

        if g_opts.lstm == false then
            g_speaker_model:backward(speaker.map[t], {grad_symbol_logp, grad_baseline})
        else
            g_speaker_model:backward({speaker.map[t], speaker.hid[t-1],speaker.cell[t-1]},
                                {grad_symbol_logp, grad_baseline, grad_hid, grad_cell})
            grad_hid = g_speaker_modules['prev_hid'].gradInput:clone()
            grad_cell = g_speaker_modules['prev_cell'].gradInput:clone()
        end
    end
    print(g_speaker_paramdx:norm())

    
    stat.reward = reward_sum:sum()
    stat.success = success:sum()
    stat.count = g_opts.batch_size
    return stat
end

local function speaker_train_batch_thread(opts_orig, speaker_paramx_orig, task_id)
    g_opts = opts_orig
    g_speaker_paramx:copy(speaker_paramx_orig)
    local stat = speaker_train_batch(task_id)
    return g_speaker_paramdx, stat
end

-- EVERYTHING ABOVE RUNS ON THREADS

function speaker_train(N)
	local threashold = 30
    for n = 1, N do
        epoch_num = n
        local stat = {} --for the epoch
        local x = g_speaker_paramx:clone()
		for k = 1, g_opts.nbatches do
            xlua.progress(k, g_opts.nbatches)
            batch_num = k
            local task_id = 1 --all game in a batch share the same task
            if g_opts.nworker > 1 then
                g_speaker_paramdx:zero()
                for w = 1, g_opts.nworker do
                    g_workers:addjob(w, speaker_train_batch_thread,
                        function(speaker_paramdx_thread, s)
                            g_speaker_paramdxadd(speaker_paramdx_thread)
                            merge_stat(stat, s)
                        end, g_opts, g_speaker_paramx, task_id)
                end
                g_workers:synchronize()
            else
                local s = speaker_train_batch(task_id)
                merge_stat(stat, s)
            end
            
            g_update_speaker_param(g_speaker_paramx, g_speaker_paramdx, task_id)
        end
        local x_p =  g_speaker_paramx:clone()
        print((x-x_p):norm())
        print(' ')
        
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
            g_opts.save = 'model_at'..threashold
            g_save_model()
            threashold  = threashold + 10
        end
        g_opts.save = 'model_epoch'
        g_save_model()

        stat.epoch = #g_log + 1
        print(format_stat(stat))
        table.insert(g_log, stat)
    end

end
