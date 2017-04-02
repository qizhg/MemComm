require 'optim'
require('nn')
require('nngraph')
--g_disp = require('display')

function run(task_id)
	local batch = batch_init(g_opts.batch_size, task_id)
	
    -- record the episode
    local active = {}
    local reward = {}
    local action = {}
    local baseline = {}


    local speaker = {}
    speaker.map = {}
    speaker.hid = {} 
    speaker.cell = {}
    speaker.hid[0] = torch.Tensor(#batch, g_opts.hidsz):fill(0)
    speaker.cell[0] = torch.Tensor(#batch, g_opts.hidsz):fill(0)

    --play the game (forward pass)
    local a = batch_active(batch)
    local map = batch_speaker_map(batch,a)
    local objmap = map[1][4]
    local dst = {}
    for y = 1, g_opts.map_height do
        for x = 1, g_opts.map_width do
            if objmap[y][x]==1 then 
                dst.y=y
                dst.x=x
            end
        end
    end 

    local agent = batch[1].listener
    print('dst at '..dst.y..'  '..dst.x)
    for t = 1, g_opts.max_steps do
        print('agent at '..agent.loc.y..'  '..agent.loc.x)

        active[t] = batch_active(batch)

    	speaker.map[t] = batch_speaker_map(batch,active[t])
        local speaker_out
        if g_opts.lstm == false then
            speaker_out = g_speaker_model:forward(speaker.map[t])
            ----speaker_out  = {symbols_logprob, symbol_baseline}
        else
            speaker_out = g_speaker_model:forward({speaker.map[t], speaker.hid[t-1],speaker.cell[t-1]})
            ----speaker_out  = {symbols_logprob, symbol_baseline, hidstate, cellstate}
            speaker.hid[t] = speaker_out[3]:clone()
            speaker.cell[t] = speaker_out[4]:clone()
        end

        baseline[t] = speaker_out[2]:clone():cmul(active[t])
        action[t] = sample_multinomial(torch.exp(speaker_out[1])) --(#batch, 1)
        print(torch.exp(speaker_out[1])[1]:view(1,-1))


        batch_listener_act(batch, action[t], active[t])
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t])
        if batch[1].finished == true then
            print('!!!')
        end

    end
    io.read()
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
    local grad_hid = torch.Tensor(#batch, g_opts.hidsz):fill(0)
    local grad_cell = torch.Tensor(#batch, g_opts.hidsz):fill(0)

    g_speaker_paramdx:zero()
    local reward_sum = torch.Tensor(#batch):zero() --running reward sum
    local norm = 0
    for t = g_opts.max_steps, 1, -1 do
        reward_sum:add(reward[t])

        local speaker_out
        if g_opts.lstm == false then
            speaker_out =g_speaker_model:forward(speaker.map[t])
            ----  speaker_out  = {symbol_logprob, symbol_baseline}
        else
            speaker_out = g_speaker_model:forward({speaker.map[t], speaker.hid[t-1],speaker.cell[t-1]})
        end
         ---- compute speaker_grad baseline
        local R = reward_sum:clone() --(#batch, )
        R:cmul(active[t]) --(#batch, )
        local grad_baseline = g_speaker_bl_loss:backward(baseline[t], R):mul(g_opts.alpha):div(#batch/2) --(#batch, 1)
    
        ----  compute speaker_grad_action via GAE
        local grad_symbol_logp = torch.Tensor(#batch, g_opts.num_symbols):zero()
        grad_symbol_logp:scatter(2, action[t], A_GAE[t]:view(-1,1):neg())

        ----  compute speaker_grad_action with entropy regularization
        local num_batchs = (epoch_num-1)*g_opts.nbatches + batch_num
        local beta = g_opts.beta_start - num_batchs*g_opts.beta_start/g_opts.beta_end_batch
        beta = math.max(0.001,beta)
        local logp = speaker_out[1]
        local entropy_grad = logp:clone():add(1)
        entropy_grad:cmul(torch.exp(logp))
        entropy_grad:mul(beta)
        entropy_grad:cmul(active[t]:view(-1,1):expandAs(entropy_grad):clone())
        grad_symbol_logp:add(entropy_grad)
        grad_symbol_logp:div(#batch/2)
        if g_opts.lstm == false then
            g_speaker_model:backward(speaker.map[t], {grad_symbol_logp, grad_baseline})
        else
            g_speaker_model:backward({speaker.map[t], speaker.hid[t-1],speaker.cell[t-1]},
                                {grad_symbol_logp, grad_baseline, grad_hid, grad_cell})
            grad_hid = g_speaker_modules['prev_hid'].gradInput:clone()
            grad_cell = g_speaker_modules['prev_cell'].gradInput:clone()
        end
    end
    --print(g_speaker_paramdx:norm())

    local stat={}
    stat.reward = reward_sum:sum()
    stat.success = success:sum()
    stat.count = g_opts.batch_size
    return stat
end