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

function batch_speaker_map(batch, active, t)
    local num_channels = 3 + g_opts.num_types_objects * 2
    local map = torch.Tensor(#batch, num_channels, g_opts.map_height, g_opts.map_width)    
    map:fill(0)
    for i, g in pairs(batch) do
        if active[i] == 1 then
            map[i] = g:to_fullmap_obs()
        end
    end
    return map
end

function batch_listener_localmap(batch, active)
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
    local reward = torch.Tensor(#batch):zero()
    for i, g in pairs(batch) do
        if active[i] == 1 then
            reward[i] = g:get_reward(is_last)
        end
    end
    return reward:view(-1)
end

function batch_update(batch, active)
    for i, g in pairs(batch) do
        if active[i] == 1 then
            g:update()
        end
    end
end
