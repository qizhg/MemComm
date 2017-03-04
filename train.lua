

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
    obs:fill(g_vocab['nil'])


    --play the game (forward pass)
    for t = 1, g_opts.max_steps do
        --get active games
        active[t] = batch_active(batch)
    	--get the latest observation
    	obs[t] = batch_obs(batch,active)
        --get input
        local mem_input = torch.Tensor(g_opts.memsize, #batch, in_dim)
        mem_input:fill(g_vocab['nil'])
        local mem_start, mem_end
        mem_start = math.max(1, t - g_opts.memsize +1)
        mem_end = t
        mem_input[{{1,mem_end-mem_start+1}}] = obs[{{mem_start,mem_end}}]
        local last_obs = mem_input[1]:clone() --(#batch, in_dim)
        mem_input = mem_input:transpose(1,2) --(#batch, memsize, in_dim)
        

        --forward input to get output = {action_logprob, baseline}
        local out = g_model:forward({mem_input, last_obs})  --out[1] = action_logprob
        action[t] = sample_multinomial(torch.exp(out[1]))  --(#batch, 1)
        
        
        --act & update
        batch_act(batch, action[t], active)
        batch_update(batch, active[t])

        --get reward
        reward[t] = batch_reward(batch, active[t]) --(#batch, )
    end
end

function train(N)
	--N: number of epochs
	--nbatches: number of batches in an epoch
	--batch_size: number of games in a batch 
	for n = 1, N do
		for k=1, g_opts.nbatches do
            print(k)
			train_batch() --get g_paramx, g_paramdx
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