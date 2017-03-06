-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

function batch_init(size)
    local batch = {}
    for i = 1, size do
        batch[i] = new_game()
    end
    return batch
end

function batch_active(batch)
    local active = torch.Tensor(#batch):zero()
    for i, g in pairs(batch) do
        if g:is_active() then
            active[i] = 1
        end
    end
    return active:view(-1)
end

function batch_obs(batch, active) --return 0-1 vector

    local input = torch.Tensor(#batch, 2*g_opts.visibility+1, 2*g_opts.visibility+1, g_opts.nwords)
    input:fill(g_vocab['nil'])
    for i, g in pairs(batch) do
        if active[i] == 1 then 
            g:get_visible_state(input[i])
        end
    end
    input = input:view(#batch, -1) --(#batch,in_dim)
    return input
end


function batch_act(batch, action, active)
    --active = active:view(#batch, g_opts.nagents)
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            --set_current_agent(g, a)
            if active[i] == 1 then
                g:act(action[i][a])
            end
        end
    end
end

function batch_update(batch, active)
    --active = active:view(#batch, g_opts.nagents)
    for i, g in pairs(batch) do
        if active[i] == 1 then
            g:update()
        end
    end
end


function batch_reward(batch, active, is_last)
    --active = active:view(#batch, g_opts.nagents)
    local reward = torch.Tensor(#batch):zero()
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            set_current_agent(g, a)
            if active[i] == 1 then
                reward[i] = g:get_reward(is_last)
            end
        end
    end
    return reward:view(-1)
end

function batch_terminal_reward(batch)
    local reward = torch.Tensor(#batch, g_opts.nagents):zero()
    for i, g in pairs(batch) do
        if g.get_terminal_reward then
            for a = 1, g_opts.nagents do
                set_current_agent(g, a)
                reward[i][a] = g:get_terminal_reward()
            end
        end
    end
    return reward:view(-1)
end



function batch_active(batch)
    local active = torch.Tensor(#batch, g_opts.nagents):zero()
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            set_current_agent(g, a)
            if g:is_active() then
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

function batch_success(batch)
    local success = torch.Tensor(#batch):fill(0)
    for i, g in pairs(batch) do
        if g:is_success() then
            success[i] = 1
        end
    end
    return success
end
