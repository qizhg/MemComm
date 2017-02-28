

function batch_init(size)
	local batch = {}
	for i = 1, size do
		batch[i] = new_game() --from init.lua 
	end
	return batch
end

function batch_active(batch)
	local active = torch.Tensor(#batch, g_opts.nagents):zero()
	for i, g in pairs(batch) do
		for a = 1, g_opts.nagents do
			set_current_agent(g, a)
			if g:is_active() then --should see Traffic.lua
				active[i][a] = 1
				-- this is little hacky
                if torch.type(g) == 'CombatGame' and g.agent.killed then
                    active[i][a] = 0
                end
            end
        end
	end
	return active:view(-1)
end

function batch_input(batch, active, t)
	-- only for active agents
	active = active:view(#batch, g_opts.nagents)
	local input = torch.Tensor(#batch,g_opts.nagents,2*g_opts.visibility+1, 2*g_opts.visibility+1, g_opts.nwords)
	input:fill(0)

	for i, g in pairs(batch) do
		for a = 1, g_opts.nagents do
			set_current_agent(g, a)
			if active[i][a] == 1 then
				g:get_visible_state(input[i][a])
			end
		end
	end
	input = input:view(#batch * g_opts.nagents, -1)
	return input
end

function batch_act(batch, action, active)
    active = active:view(#batch, g_opts.nagents)
    action = action:view(#batch, g_opts.nagents)
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            set_current_agent(g, a)
            if active[i][a] == 1 then
                g:act(action[i][a])
            end
        end
    end
end

function batch_reward(batch, active, is_last)
    active = active:view(#batch, g_opts.nagents)
    local reward = torch.Tensor(#batch, g_opts.nagents):zero()
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            set_current_agent(g, a)
            if active[i][a] == 1 then
                reward[i][a] = g:get_reward(is_last)
            end
        end
    end
    return reward:view(-1)
end

function batch_update(batch)
    for i, g in pairs(batch) do
        g:update()
    end
end

function batch_success(batch) -- in Trafiic, success = 0 collision
    local success = torch.Tensor(#batch):fill(0)
    for i, g in pairs(batch) do
        if g:is_success() then
            success[i] = 1
        end
    end
    return success
end

