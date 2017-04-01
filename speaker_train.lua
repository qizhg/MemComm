require 'optim'
require('nn')
require('nngraph')
--g_disp = require('display')

local function speaker_train_batch(task_id)
	local batch = batch_init(g_opts.batch_size, task_id)
	
    -- record the episode
    local active = {}
    local reward = {}
    local action = {}
    local baseline = {}


    local speaker = {}
    local speaker_hidsz = g_opts.hidsz
    speaker.map = {}
    speaker.hid = {} 
    speaker.cell = {}
    speaker.hid[0] = torch.Tensor(#batch, speaker_hidsz):fill(0)
    speaker.cell[0] = torch.Tensor(#batch, speaker_hidsz):fill(0)

    --play the game (forward pass)
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)

    	speaker.map[t] = batch_speaker_map(batch,active[t])
        local speaker_out = g_speaker_model:forward(speaker.map[t])
        ----speaker_out  = {symbols_logprob, symbol_baseline}

        baseline[t] = speaker_out[2]:clone():cmul(active[t])
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
        delta[t] = reward[g_opts.max_steps] + g_opts.gamma*baseline[t+1] - baseline[t]
    end
    local A_GAE={} --GAE advatage
    A_GAE[g_opts.max_steps] = delta[g_opts.max_steps]
    for t=g_opts.max_steps-1, 1, -1 do 
        A_GAE[t] = delta[t] + g_opts.gamma*g_opts.lambda*A_GAE[t+1] 
    end


    --backward pass
    g_speaker_paramdx:zero()
    local reward_sum = torch.Tensor(#batch):zero() --running reward sum
    local norm = 0
    for t = g_opts.max_steps, 1, -1 do
        reward_sum:add(reward[t])

        --speaker
        local speaker_out =g_speaker_model:forward(speaker.map[t])
        ----  speaker_out  = {symbol_logprob, symbol_baseline}
         ---- compute speaker_grad baseline
        local R = reward_sum:clone() --(#batch, )
        R:cmul(active[t]) --(#batch, )
        local grad_baseline = g_speaker_bl_loss:backward(baseline[t], R):mul(g_opts.alpha):div(#batch/8) --(#batch, 1)
    
        ----  compute speaker_grad_action via GAE
        local grad_symbol_logp = torch.Tensor(#batch, g_opts.num_symbols):zero()
        grad_symbol_logp:scatter(2, action[t], A_GAE[t]:view(-1,1):neg())

        ----  compute speaker_grad_action with entropy regularization
        local num_batchs = (epoch_num-1)*g_opts.nbatches + batch_num
        local beta = g_opts.beta_start - num_batchs*g_opts.beta_start/g_opts.beta_end_batch
        beta = math.max(0,beta)
        local logp = speaker_out[1]
        local entropy_grad = logp:clone():add(1)
        entropy_grad:cmul(torch.exp(logp))
        entropy_grad:mul(beta)
        entropy_grad:cmul(active[t]:view(-1,1):expandAs(entropy_grad):clone())
        grad_symbol_logp:add(entropy_grad)
        grad_symbol_logp:div(#batch/8)
        g_speaker_model:backward(speaker.map[t],
                                {grad_symbol_logp, grad_baseline})
    end
    print(g_speaker_paramdx:norm())

    local stat={}
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
