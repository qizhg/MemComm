local JunBase, parent = torch.class('JunBase', 'MazeBase')

function JunBase:__init(opts, vocab)
    parent.__init(self, opts, vocab)
    self:add_default_items() -- blocks, waters

    --add agents
    ----speaker
    self.num_symbols = opts.num_symbols
    self.speaker = self:add_item({type = 'agent', name = 'speaker'}) --move actions
    self:set_speaker_actions() --delete move actions, add talk actions
    ----listener
    self.listener = self:place_item_rand({type = 'agent', name = 'listener'}) --move actions
    self.listener_visibility = opts.listener_visibility
    self.pickup_enable = opts.pickup_enable
    if self.pickup_enable ==true then
        self:add_listener_pickup_actions() --pick up actions
    end
    self.agent = self.listener --for calling parent

    
    --add random objects
    self.num_types_objects = opts.num_types_objects
    self.min_num_objects = opts.min_num_objects
    self.max_num_objects = opts.max_num_objects
    self:add_objects()

    --items type dict
    self.id2ItemType = {}
    self.ItemType2id = {}
    self.id2ItemType[1] =  'agent' 
    self.id2ItemType[2] =  'block'
    self.id2ItemType[3] =  'water'
    self.ItemType2id.agent =  1
    self.ItemType2id.block = 2
    self.ItemType2id.water = 3
    for obj_id = 1, self.num_types_objects do
        self.id2ItemType[3+obj_id] =  'obj'.. obj_id
        self.ItemType2id['obj'.. obj_id] = 3 + obj_id
    end

    --task finished dict
    self.finished = false
    self.id2task = {}
    self.task2id = {}
    for obj_id = 1, self.num_types_objects do
        self.id2task[obj_id] = 'visit '..'obj'..obj_id
        self.task2id['visit '..'obj'..obj_id] = 2*obj_id-1

        self.id2task[self.num_types_objects+obj_id] = 'pickup '..'obj'..obj_id
        self.task2id['pickup '..'obj'..obj_id] = self.num_types_objects+obj_id

    end
    self.num_tasks = #self.id2task
    self.task_id = torch.random(1, self.num_tasks)
end


function JunBase:add_objects()
    local num_objects = torch.LongTensor(self.num_types_objects)
    num_objects:random(self.min_num_objects,self.max_num_objects)
    for obj_id = 1, self.num_types_objects do
        for i = 1, num_objects[obj_id] do
            self:place_item_rand({type = 'obj'.. obj_id, picked_up = false})
        end
    end
    --self:place_item({type = 'obj'.. 1, picked_up = false},self.listener.loc.y-1,self.listener.loc.x)

end

function JunBase:add_listener_pickup_actions()

    self.listener:add_action('pickup_up',
        function(self) --self for litsener
            local e = self.map:loc_pickup(self.loc.y - 1, self.loc.x)
            if e then
                e.attr.picked_up = true
            end
        end)
    self.listener:add_action('pickup_down',
        function(self) --self for litsener
            local e = self.map:loc_pickup(self.loc.y + 1, self.loc.x)
            if e then
                e.attr.picked_up = true
            end
        end)
    self.listener:add_action('pickup_left',
        function(self) --self for litsener
            local e = self.map:loc_pickup(self.loc.y, self.loc.x - 1)
            if e then
                e.attr.picked_up = true
            end
        end)
    self.listener:add_action('pickup_right',
        function(self) --self for litsener
            local e = self.map:loc_pickup(self.loc.y, self.loc.x + 1)
            if e then
                e.attr.picked_up = true
            end
        end)
end

function JunBase:set_speaker_actions()
    --remove all actions
    self.speaker.action_names = {}  -- id -> name
    self.speaker.action_ids = {}    -- name -> id
    self.speaker.actions = {}       -- id -> func
    self.speaker.nactions = 0

    for s = 1, self.num_symbols do
        self.speaker:add_action('talk_'..s,
            function(self) --self for speaker
                self.maze.listener.attr.talk = s
            end)
    end


end

-- 3D map representation for conv model 
function JunBase:to_fullmap_obs()
    local num_channels = 3 + self.num_types_objects * 2 --3: block, water, listener
    local map = torch.Tensor(num_channels,self.map.height, self.map.width)
    map:fill(0)
    for y = 1, self.map.height do
        for x = 1, self.map.width do
            for _,e in ipairs(self.map.items[y][x]) do
                local itemtype_id = self.ItemType2id[e.type]
                if itemtype_id <= 3 then
                    map[itemtype_id][y][x] = 1
                elseif e.picked_up == false then
                    local obj_id = itemtype_id -3
                    map[3 + obj_id ][y][x] = 1
                else
                    local obj_id = itemtype_id -3
                    map[3 + self.num_types_objects + obj_id][y][x] = 1
                end
            end
        end
    end
    return map
end

-- 3D map representation for conv model 
function JunBase:to_localmap_obs(agent, visibility)
    local agent = agent or self.listener
    local visibility = visibility or self.listener_visibility

    local num_channels = 3 + self.num_types_objects * 2 --3: block, water, listener
    local local_map = torch.Tensor(num_channels,visibility*2 + 1, visibility*2 + 1)
    local_map:fill(0)
    local yy,xx
    yy = 0
    for y = math.max(1, agent.loc.y - visibility) , math.min(self.map.height, agent.loc.y + visibility) do
        yy = yy + 1
        xx = 0
        for x = math.max(1, agent.loc.x - visibility) , math.min(self.map.width, agent.loc.x + visibility) do
            xx = xx + 1
            for _,e in ipairs(self.map.items[y][x]) do
                local itemtype_id = self.ItemType2id[e.type]
                if itemtype_id <= 3 then
                    local_map[itemtype_id][yy][xx] = 1
                elseif e.picked_up == false then
                    local obj_id = itemtype_id -3
                    local_map[3 + obj_id  ][yy][xx] = 1
                else
                    local obj_id = itemtype_id -3
                    local_map[3 + self.num_types_objects + obj_id][yy][xx] = 1
                end
            end
        end
    end
    return local_map
end

function JunBase:update()
    
    parent.update(self) -- t = t+1 
    
    --picked up items follow the listener
    for i = 1, #self.items do
        local e = self.items[i]
        if e.attr.picked_up == true then
            self.map:remove_item(e)
            e.loc.y = self.listener.loc.y
            e.loc.x = self.listener.loc.x
            self.map:add_item(e)
        end
    end

    --check for task finished
    local task_obj_id = self.task_id % self.num_types_objects
    local items = self.map.items[self.listener.loc.y][self.listener.loc.x]
    for i = 1, #items do
        if items[i] == 'obj'.. task_obj_id then
            if self.task_id / self.num_types_objects == 0 and items.picked_up == false then --visit
                self.finished = true
            elseif self.task_id / self.num_types_objects == 1 and items.picked_up == true then --pick_up
                self.finished = true
            end
        end
    end


end

function JunBase:listener_act(action)
    self.listener:act(action)
end

function JunBase:get_reward()
    if self.finished then
        return -self.costs.goal
    else
        return parent.get_reward(self)
    end
end