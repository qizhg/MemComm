-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

local Traffic, parent = torch.class('Traffic', 'MazeBase')

function Traffic:__init(opts, vocab)
    parent.__init(self, opts, vocab)

    self.add_rate = opts.add_rate
    self.add_block = opts.add_block
    self.max_agents = opts.max_agents
    self.costs.collision = self.costs.collision
    self.action_delay = opts.action_delay or 0

    self.source_locs = {}
    self.dest_locs = {}
    self.routes = {}
    self:build_roads()

    self.agents = {}
    self.agents_inactive = {}
    self.agents_active = {}
    self.nagents = opts.nagents
    for i = 1, self.nagents do
        local agent = self:place_item({type = 'agent', 
            _name = 'agent' .. i, _ascii = '@' .. i, _ind = i}, 1, 1)      
        agent.attr._invisible = true
        local colors = {'red', 'green', 'yellow', 'blue', 'magenta', 'cyan'}
        agent.attr._ascii_color = { colors[torch.random(#colors)] }
        --agent.abs_loc_visible = true
        agent.active = false
        agent.act = function(self, action_id)
            assert(self.active)
            MazeAgent.act(self, action_id)
        end

        self.agents[i] = agent
        self.agents_inactive[i] = agent
    end
    self.agent = self.agents[1]
    self.ncollision_total = 0
    self:add_agent()
end

function Traffic:add_agent()
    
    if #self.agents_active >= 1 then
        self.add_rate = 0.0
    else
        self.add_rate = 1.0
    end
    
    --local src = self.source_locs[torch.random(#self.source_locs)]
    local src_id = torch.random(1)
    local src = self.source_locs[src_id]
    
    if #self.agents_active >= self.max_agents then
        return
    end
    local ri = src.routes[torch.random(#src.routes)]
    self:place_item({type = 'route'..ri}, self.length + 1, self.length + 1)
    local route = self.routes[ri]
    if torch.uniform() < self.add_rate then
        if #self.agents_inactive == 0 then
            return
        end
        local r = torch.random(#self.agents_inactive)
        local agent = self.agents_inactive[r]
        self.map:remove_item(agent)
        agent.loc.y = src.y
        agent.loc.x = src.x
        table.remove(self.agents_inactive, r)
        agent.active = true
        agent.attr._invisible = false
        agent.t = 0
        agent.src_id = src_id
        agent.route = route
        agent.route_pos = 1
        agent.attr._route = 'route' .. ri
        self.map:add_item(agent)
        table.insert(self.agents_active, agent)        
        -- agent.attr._ascii = agent.attr._ind .. ri
        agent.attr._ascii = '<>'
     end
    
end

function Traffic:update()
    parent.update(self)

    self.success_pass = 0
    self.ncollision = 0
    for _, agent in pairs(self.agents) do
        agent.success_pass = 0
        --agent.ncollision = 0
    end
    local t = {}
    for _, agent in pairs(self.agents_active) do
        agent.t = agent.t + 1

        --[[
        local src = self.source_locs[agent.src_id]
        local reach_some_dst = false
        for route_id = 1, #src.routes do
            
            local dst = self.routes[route_id][#self.routes[route_id]
            if agent.loc.y == dst.y and agent.loc.x == dst.x then
                reach_some_dst = true
                if agent.route[#agent.route].y == dst.y and agent.route[#agent.route].x == dst.x then
                    agent.success_pass = agent.success_pass + 1
                    self.success_pass = self.success_pass + 1
                else
                    agent.success_pass = agent.success_pass - 1
                    self.success_pass = self.success_pass - 1
                end
                agent.attr._invisible = true
                agent.active = false
                table.insert(self.agents_inactive, agent)
                self.map:remove_item(agent)
                agent.loc.y = 1
                agent.loc.x = 1
                self.map:add_item(agent)
            end
        end
        if reach_some_dst == false then 
            table.insert(t, agent)
        end
        --]]
        
        
    
        
        local dst = agent.route[#agent.route]
        if agent.loc.y == dst.y and agent.loc.x == dst.x then
            agent.success_pass = agent.success_pass + 1
            self.success_pass = self.success_pass + 1
            agent.attr._invisible = true
            agent.active = false
            table.insert(self.agents_inactive, agent)
            self.map:remove_item(agent)
            agent.loc.y = 1
            agent.loc.x = 1
            self.map:add_item(agent)
        else
            table.insert(t, agent)
        end
        
        
        
    end
    self.agents_active = t

end

function Traffic:get_reward(is_last)

    local r = 0
    r = r - self.agent.success_pass * self.costs.pass
    r = r - self.agent.ncollision * self.costs.collision
    --r = r - self.agent.t * self.costs.wait
    r = r - self.costs.wait
    --r = r - self:ManhattanDis2dst() * self.costs.distance
    return r
end

function Traffic:ManhattanDis2dst()
    local agent = self.agent
    local dst = agent.route[#agent.route]
    local Mdistance = math.abs(agent.loc.y - dst.y) + math.abs(agent.loc.x - dst.x)
    return Mdistance
end

function Traffic:is_active()
    return self.agent.active
end

function Traffic:is_success()
    if self.agent.success_pass < 1 then
        return false
    else
        return true
    end
end
