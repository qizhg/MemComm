-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

paths.dofile('MazeItem.lua')
paths.dofile('MazeAgent.lua')
paths.dofile('MazeMap.lua')

local MazeBase = torch.class('MazeBase')

function MazeBase:__init(opts, vocab)
    self.map = MazeMap(opts)
    self.visibility = opts.visibility or 1
    self.max_attributes = opts.max_attributes
    self.vocab = vocab
    self.t = 0
    self.costs = {}
    for i, j in pairs(opts.costs) do
        self.costs[i] = j
    end

    -- This list contains EVERYTHING in the game.
    self.items = {}
end

function MazeBase:add_prebuilt_item(e)
    e.id = #self.items+1
    self.items[#self.items+1] = e
    if e.loc then
        self.map:add_item(e)
    end
    return e
end

function MazeBase:add_item(attr)
    local e
    if attr._factory then
        e = attr._factory(attr,self)
    else
        if attr.type == 'agent' then
            e = MazeAgent(attr, self)
        else
            e = MazeItem(attr)
        end
    end
    self:add_prebuilt_item(e)
    return e
end

function MazeBase:place_item(attr, y, x)
    attr.loc = {y = y, x = x}
    return self:add_item(attr)
end

function MazeBase:remove(item, l)
    for i = 1, #l do
        if l[i] == item then
            table.remove(l, i)
            break
        end
    end
end

function MazeBase:remove_item(item)
    if item.loc then
        self.map:remove_item(item)
    end
    if item.type then
        self:remove(item, self.items_bytype[item.type])
    end
    if item.name then
        self.item_byname[item.name] = nil
    end
    self:remove(item, self.items)
end

-- Agents call this function to perform action
function MazeBase:act(action)
    self.agent:act(action)
end

-- Update map state after each step
function MazeBase:update()
    self.t = self.t + 1
    for i = 1, #self.items do
        if self.items[i].updateable then
            self.items[i]:update(self)
        end
    end
    if self.finish_by_goal then
        local items = self.map.items[self.agent.loc.y][self.agent.loc.x]
        for i = 1, #items do
            if items[i].type == 'goal' then
                self.finished = true
            end
        end
    end
end

function MazeBase:to_sentence_item(e, sentence)
    local s = e:to_sentence(self.agent.loc.y, self.agent.loc.x)
    if g_opts.batch_size == 1 then
        print(self.agent.name, ':', table.concat(s,', '))
    end
    for i = 1, #s do
        sentence[i] = self.vocab[s[i]]
    end
end

-- Tensor representation that can be feed to a model
function MazeBase:to_sentence(sentence)
    local count=0
    local sentence = sentence or torch.Tensor(#self.items, self.max_attributes):fill(self.vocab['nil'])
    for i = 1, #self.items do
        if not self.items[i].attr._invisible then
            count= count + 1
            if count > sentence:size(1) then error('increase memsize!') end
            self:to_sentence_item(self.items[i], sentence[count])
        end
    end
    return sentence
end

function MazeBase:get_visible_state(data, use_lut)

    for dy = -self.visibility, self.visibility do
        for dx = -self.visibility, self.visibility do
            local y, x
            y = self.agent.loc.y + dy
            x = self.agent.loc.x + dx
            if self.map.items[y] and self.map.items[y][x] then
                for _, e in pairs(self.map.items[y][x]) do
                    if self.agent == e or (not e.attr._invisible) then
                        local s = e:to_sentence(0, 0, true)
                        --if g_opts.batch_size == 1 then
                        --    print(self.agent.name, ':', table.concat(s,', '))
                        --end
                        for i = 1, #s do
                            if self.vocab[s[i]] == nil then error('not found in dict:' .. s[i]) end
                            local data_y = dy + self.visibility + 1
                            local data_x = dx + self.visibility + 1
                            data[data_y][data_x][self.vocab[s[i]]] = 1
                        end
                    end
                end
            end
        end
    end
end

-- This reward signal is used for REINFORCE learning
function MazeBase:get_reward(is_last)
    local items = self.map.items[self.agent.loc.y][self.agent.loc.x]
    local reward = -self.costs.step
    for i = 1, #items do
        if items[i].type ~= 'agent' then
            if items[i].reward then
                reward = reward + items[i].reward
            elseif self.costs[items[i].type] then
                reward = reward - self.costs[items[i].type]
            end
        end
    end
    return reward
end

function MazeBase:is_active()
    return (not self.finished)
end

function MazeBase:is_success()
    if self:is_active() then
        return false
    else
        return true
    end
end

