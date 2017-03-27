-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function batch_init(size)
    local batch = {}
    for i = 1, size do
        batch[i] = g_mazebase.new_game()
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

function batch_listener_localmap(batch, active, t)
    local num_channels = 3 + g_opts.num_types_objects * 2
    local visibility = g_opts.listener_visibility
    local localmap = torch.Tensor(#batch, num_channels, visibility*2+1, visibility*2+1)
    localmap:fill(0)
    for i, g in pairs(batch) do
        if active[i] == 1 then
            localmap[i] = g:to_localmap_obs() --default:listener's local map
        end
    end
    return localmap
end

function batch_listener_act(batch, listener_action, active)
    for i, g in pairs(batch) do
        if active[i] == 1 then
            g:listener_act(listener_action[i])
        end
    end
end

function batch_reward(batch, active, is_last)
    active = active:view(#batch, g_opts.nagents)
    local reward = torch.Tensor(#batch, g_opts.nagents):zero()
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            g.agent = g.agents[a]
            if active[i][a] == 1 then
                reward[i][a] = g:get_reward(is_last)
            end
        end
    end
    return reward:view(-1)
end

function batch_update(batch, active)
    active = active:view(#batch, g_opts.nagents)
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            g.agent = g.agents[a]
            if active[i][a] == 1 then
                g:update()
                break
            end
        end
    end
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
