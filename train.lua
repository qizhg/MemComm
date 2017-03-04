

require'optim'

function train_nonbatch()
	---get a new game
	local game = new_game()

	-- record the episode
    local reward = {}
    local obs = {}
    local action = {}

    --
    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords
    local model_input = torch.Tensor(g_opts.memsize,in_dim):fill(game.vocab['nil'])


    --play the game
    for t = 1, g_opts.max_steps do
    	--get mem
    	obs[t] = nonbatch_obs(game)
    	model_input[{{1,-2}}] = model_input[{{2,-1}}]
    	model_input[-1] = obs[t]:clone()
    	
        --get context MQN



    end
end

function train(N)
	--N: number of epochs
	--nbatches: number of batches in an epoch
	--batch_size: number of games in a batch 
	for n = 1, N do
		for k=1, g_opts.nbatches do
			train_nonbatch() --get g_paramx, g_paramdx
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