
require'optim'

function train_batch()
    local batch = batch_init(g_opts.batch_size)

    local active = {}
    local reward = {}
    local action = {}

    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords
    local obs = torch.Tensor(g_opts.max_steps, g_opts.batch_size * g_opts.nagents, in_dim)
    obs:fill(0)

    -- communication states
    local comm_state = {}
    comm_state[1] = torch.zeros(g_opts.batch_size * g_opts.nagents, g_opts.nagents, g_opts.hidsz)
    local comm_mask = {}
    local comm_mask_default = torch.ones(g_opts.nagents, g_opts.nagents)
    for s = 1, g_opts.nagents do
        for d = 1, g_opts.nagents do
            if s == d then
                -- no self talking
                comm_mask_default[s][d] = 0
            end
        end
    end

    -- forward pass to play the games
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)
        obs[t] = batch_input(batch,active[t])

        --form input_table
        local prev_obs = torch.Tensor(g_opts.memsize, g_opts.batch_size * g_opts.nagents, in_dim):fill(0)
        if t > 1 then 
            local mem_start, mem_end
            mem_start = math.max(1, t - g_opts.memsize)
            mem_end = math.max(1, t - 1)
            prev_obs[{{1,mem_end-mem_start+1}}] = obs[{{mem_start,mem_end}}]
        end
        prev_obs = prev_obs:transpose(1,2) --(#batch*nagents, memsize, in_dim)
        ---- add time feature
        local time_feature = torch.Tensor(g_opts.batch_size * g_opts.nagents, g_opts.memsize, g_opts.memsize)
        for i = 1, g_opts.batch_size * g_opts.nagents do time_feature[i] = torch.eye(g_opts.memsize) end
        prev_obs = torch.cat(prev_obs,time_feature,3) --(#batch*nagents, memsize, in_dim+memsize)
        
        local cur_obs = obs[t]:clone() --(#batch*nagents, in_dim)

        local comm_in = comm_state[t]

        --forward 
        local out = g_model:forward({prev_obs, cur_obs, comm_in})  --out={action_logprob,baseline,comm_out}

        --take action and get reward
        action[t] = sample_multinomial(torch.exp(out[1]))
        batch_act(batch, action[t]:view(-1), active[t])
        batch_update(batch)
        reward[t] = batch_reward(batch, active[t])

        --form new comm_state
        ----form mask
        local m = comm_mask_default:view(1, g_opts.nagents, g_opts.nagents)
        m = m:expand(g_opts.batch_size, g_opts.nagents, g_opts.nagents):clone()
        ----inactive agents don;t communicate
        local m2 = active[t]:view(g_opts.batch_size, g_opts.nagents, 1):clone()
        m2 = m2:expandAs(m):clone()
        m:cmul(m2)
        m:cmul(m2:transpose(2,3))
        ----average 
        m:cdiv(m:sum(2):expandAs(m):clone():add(m:eq(0):float()))
        comm_mask[t] = m
        ----apply mask
        local h = out[3]:clone()
        h = h:view(g_opts.batch_size, g_opts.nagents, g_opts.nagents, g_opts.hidsz)
        m = m:view(g_opts.batch_size, g_opts.nagents, g_opts.nagents, 1)
        m = m:expandAs(h):clone()
        h:cmul(m)
        comm_state[t+1] = h:transpose(2,3):clone()
        comm_state[t+1]:resize(g_opts.batch_size * g_opts.nagents, g_opts.nagents, g_opts.hidsz)
    end
    
    -- backward pass
    g_paramdx:zero()
    local grad_comm = torch.zeros(g_opts.batch_size * g_opts.nagents, g_opts.nagents, g_opts.hidsz)
    local reward_sum = torch.Tensor(g_opts.batch_size * g_opts.nagents):zero() --running reward sum

    for t = g_opts.max_steps, 1, -1 do
        reward_sum:add(reward[t])

        --do one-step forward
        local prev_obs = torch.Tensor(g_opts.memsize, g_opts.batch_size * g_opts.nagents, in_dim):fill(0)
        if t > 1 then 
            local mem_start, mem_end
            mem_start = math.max(1, t - g_opts.memsize)
            mem_end = math.max(1, t - 1)
            prev_obs[{{1,mem_end-mem_start+1}}] = obs[{{mem_start,mem_end}}]
        end
        prev_obs = prev_obs:transpose(1,2) --(#batch*nagents, memsize, in_dim)
        local time_feature = torch.Tensor(g_opts.batch_size * g_opts.nagents, g_opts.memsize, g_opts.memsize)
        for i = 1, g_opts.batch_size * g_opts.nagents do time_feature[i] = torch.eye(g_opts.memsize) end
        prev_obs = torch.cat(prev_obs,time_feature,3) --(#batch*nagents, memsize, in_dim+memsize)
    
        local cur_obs = obs[t]:clone() --(#batch*nagents, in_dim)
        local comm_in = comm_state[t]
        local out = g_model:forward({prev_obs, cur_obs, comm_in})  --out={action_logprob,baseline,comm_out}

        --compute grad
        ---- each agent receive the same team reward
        local R = reward_sum:clone()
        R = R:view(#batch, g_opts.nagents)
        R = R:mean(2):expandAs(R):clone()
        R = R:view(-1, 1) 
        R:cmul(active[t]) --(#batch*g_opts.nagents, 1)
        ---- compute grad_baseline
        local baseline = out[2] --(#batch  * nagents, 1)
        baseline:cmul(active[t]) --(#batch  * nagents, 1)
        local grad_baseline = g_bl_loss:backward(baseline, R):mul(g_opts.alpha) --(#batch  * nagents, 1)
        ---- compute grad_action
        local grad_action = torch.Tensor(g_opts.batch_size * g_opts.nagents, g_opts.nactions):zero()
        local R_action = baseline - R
        grad_action:scatter(2, action[t], R_action)
        grad_action:div(g_opts.batch_size)
        ----backward and compute grad_comm
        g_model:backward({prev_obs, cur_obs, comm_in},{grad_action, grad_baseline, grad_comm})
        if t> 1 then 
            local h = g_modules['comm_in'].gradInput:clone()
            h = h:view(g_opts.batch_size, g_opts.nagents, g_opts.nagents, g_opts.hidsz)
            grad_comm = h:transpose(2,3):clone()
            --apply mask
            local m = comm_mask[t-1]
            m = m:view(g_opts.batch_size, g_opts.nagents, g_opts.nagents, 1)
            m = m:expandAs(grad_comm):clone()
            grad_comm:cmul(m)
            grad_comm:resize(g_opts.batch_size * g_opts.nagents, g_opts.nagents, g_opts.hidsz)
        end
    end

end

function train(N)
    for n = 1, N do
        local stat = {} --for the epoch
        for k = 1, g_opts.nbatches do
            local s = train_batch() --get g_paramx, g_paramdx
            g_update_param(g_paramx, g_paramdx)
        end
    end
end


function g_update_param(x, dx)
    dx:div(g_opts.nworker)
    if g_opts.max_grad_norm > 0 then
        if dx:norm() > g_opts.max_grad_norm then
            dx:div(dx:norm() / g_opts.max_grad_norm)
        end
    end
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

    if g_opts.encoder_lut then
        -- zero NIL embedding
        g_modules['encoder_lut'].weight[g_opts.encoder_lut_nil]:zero()
    end
end