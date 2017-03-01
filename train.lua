require'optim'

function train_batch()
	local batch = batch_init(g_opts.batch_size) --see batch.lua

	-- record episode states
    local reward = {}
    local input = {}
    local action = {}
    local active = {} --active agents

    --rnn hid_state
    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords
    local hid_state = torch.zeros(g_opts.batch_size * g_opts.nagents, g_opts.hidsz)
    local hid_grad = torch.zeros(g_opts.batch_size * g_opts.nagents, g_opts.hidsz)
    if g_opts.recurrent then
        hid_state:fill(g_opts.init_hid) --init_hid = 0.1 (default)
    end

    --rnn comm state
    local comm_state = torch.zeros(g_opts.batch_size * g_opts.nagents, g_opts.nagents, g_opts.hidsz)
    local comm_grad = torch.zeros(g_opts.batch_size * g_opts.nagents, g_opts.nagents, g_opts.hidsz)
	
	local comm_mask_default = torch.ones(g_opts.nagents, g_opts.nagents)--???
    local comm_mask = {}--???
    for s = 1, g_opts.nagents do
        for d = 1, g_opts.nagents do
            if s == d then
                -- no self talking
                comm_mask_default[s][d] = 0
            end
        end
    end

    -- play an episode
    for t = 1, g_opts.max_steps * g_opts.nhop do
    	--find active agents
    	active[t] = batch_active(batch) --0-1 vector of (#batch * g_opts.nagents,)
        
    	--inputs
    	input[t] = {}
    	----input
        local x = batch_input(batch, active[t], t) --(#batch * g_opts.nagents, -1)
        input[t][g_model_inputs['input']] = x
        ----hid_state
        input[t][g_model_inputs['prev_hid']] = hid_state
        ----comm
        if g_opts.comm then
            input[t][g_model_inputs['comm_in']] = comm_state
        end

        --forward pass
        local out = g_model:forward(input[t]) -- a table of outputs

        -- act when model hops completed
        if t % g_opts.nhop == 0 then
            action[t] = sample_multinomial(torch.exp(out[g_model_outputs['action_prob']]))
            batch_act(batch, action[t]:view(-1), active[t])
            batch_update(batch)
            reward[t] = batch_reward(batch, active[t], t == g_opts.max_steps * g_opts.nhop) --(#batch * g_opts.nagents,)
        end

        --update hid_state
        hid_state = out[g_model_outputs['hidstate']]:clone()

        --update comm
        if g_opts.comm then
        	-- determine which agent can talk to which agent?
            local m = comm_mask_default:view(1, g_opts.nagents, g_opts.nagents)
            m = m:expand(g_opts.batch_size, g_opts.nagents, g_opts.nagents):clone()
            ---- inactive agents don't communicate
            local m2 = active[t]:view(g_opts.batch_size, g_opts.nagents, 1):clone()
            m2 = m2:expandAs(m):clone()
            m:cmul(m2)
            m:cmul(m2:transpose(2,3))

            --average comms by dividing by number of agents
            if g_opts.comm_mode == 'avg' then
                -- average comms by dividing by number of agents
                m:cdiv(m:sum(2):expandAs(m):clone():add(m:eq(0):float()))
            end
            m:div(g_opts.comm_scale_div)
            
            --write comm_mask[t]
            comm_mask[t] = m

            --apply mask
            local h = out[g_model_outputs['comm_out']]:clone()
            h = h:view(g_opts.batch_size, g_opts.nagents, g_opts.nagents, g_opts.hidsz)
            local m = comm_mask[t]
            m = m:view(g_opts.batch_size, g_opts.nagents, g_opts.nagents, 1)
            m = m:expandAs(h):clone()
            h:cmul(m)
            
            --!!! communicate is just transpose
            comm_state = h:transpose(2,3):clone()
            comm_state:resize(g_opts.batch_size * g_opts.nagents, g_opts.nagents, g_opts.hidsz)

        end
    end --episode ends

    --backward pass





end


function train(N)
	--N: number of epochs
	--nbatches: number of batches in an epoch
	--batch_size: number of games in a batch 
	for n = 1, N do
		for k=1, g_opts.nbatches do
			train_batch() --get g_paramx, g_paramdx on a batch
			g_update_param(g_paramx, g_paramdx)
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