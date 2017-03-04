-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

local MazeAgent, parent = torch.class('MazeAgent', 'MazeItem')

function MazeAgent:__init(attr, maze)
    attr.type = 'agent'
    parent.__init(self, attr)
    self.maze = maze
    self.map = maze.map

    -- List of possible actions and corresponding functions to execute
    self.action_names = {}  -- id -> name
    self.action_ids = {}    -- name -> id
    self.actions = {}       -- id -> func
    self.nactions = 0
    self:add_move_actions()
end

function MazeAgent:add_action(name, f)
    if not self.action_ids[name] then
        self.nactions = self.nactions + 1
        self.action_names[self.nactions] = name
        self.actions[self.nactions] = f
        self.action_ids[name] = self.nactions
    else
        self.actions[self.action_ids[name]] = f
    end
end

function MazeAgent:add_move_actions()
    self:add_action('up',
        function(self)
            if self.map:is_loc_reachable(self.loc.y - 1, self.loc.x) then
                self.map:remove_item(self)
                self.loc.y = self.loc.y - 1
                self.map:add_item(self)
            end
        end)
    self:add_action('down',
        function(self)
            if self.map:is_loc_reachable(self.loc.y + 1, self.loc.x) then
                self.map:remove_item(self)
                self.loc.y = self.loc.y + 1
                self.map:add_item(self)
            end
        end)
    self:add_action('left',
        function(self)
            if self.map:is_loc_reachable(self.loc.y, self.loc.x - 1) then
                self.map:remove_item(self)
                self.loc.x = self.loc.x - 1
                self.map:add_item(self)
            end
        end)
    self:add_action('right',
        function(self)
            if self.map:is_loc_reachable(self.loc.y, self.loc.x + 1) then
                self.map:remove_item(self)
                self.loc.x = self.loc.x + 1
                self.map:add_item(self)
            end
        end)
    self:add_action('stop',
        function(self)
            -- do nothing
        end)
end

-- Agents call this function to perform action
function MazeAgent:act(action_id)
    local f = self.actions[action_id]
    if f == nil then
       print('Available actions are: ')
       for k,v in pairs(self.actions) do
	  print(k)
       end
       error('Could not find action for action_id: ' .. action_id)
    end
    f(self)
    self.last_action = action_id
end