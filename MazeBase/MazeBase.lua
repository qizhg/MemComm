paths.dofile('MazeItem.lua')
paths.dofile('MazeMap.lua')
paths.dofile('MazeAgent.lua')

local MazeBase = torch.class('MazeBase')

function MazeBase:__init(opts,vocab)
	--opts:
	--vocab:

	self.map = MazeMap(opts)
	self.visibility = opts.visibility or 1 --???
	self.max_attributes = opts.max_attributes --???
	self.vocab = vocab --???
	self.t = 0  --???
	self.costs = {} --???
	self.push_action = opts.push_action --???
	self.crumb_action = opts.crumb_action --???
	self.flag_visited = opts.flag_visited --???
	self.enable_boundary = opts.enable_boundary
	self.enable_corners = opts.enable_corners --???

	self.items={} --???
	self.items_bytype = {} --???
	self.items_byname ={} --???

	if self.enable_boundary == 1 then
		self:add_boundary()
	end

	for i, j in pairs(opts.costs) do --??? why not self.costs =opts.costs
		self.costs[i] = j
	end

	self.ngoals = opts.ngoals or 1 --???
	self.nagents = opts.nagents or 1 --???
	self.nblocks = opts.nblocks or 0 --???
	self.nwater = opts.nwater or 0 --???
	self.finished = false --???
	self.finish_by_goal = false --???
end

function MazeBase:add_boundary()
    for x = 1, self.map.width do
        self:place_item({type = 'block'}, 1, x)
        self:place_item({type = 'block'}, self.map.height, x)
    end
    for y = 2, self.map.height-1 do
        self:place_item({type = 'block'}, y, 1)
        self:place_item({type = 'block'}, y, self.map.width)
    end
end

---------add items----------------------
function MazeBase:place_item(attr, y, x)
	attr.loc ={y=y,x=x}
	self:add_item(attr)
end

function MazeBase:add_item(attr)
    local e
    if attr._factory then --???
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

function MazeBase:add_prebuilt_item(e)
    e.id = #self.items+1
    self.items[#self.items+1] = e
    if not self.items_bytype[e.type] then
        self.items_bytype[e.type] = {}
    end
    table.insert(self.items_bytype[e.type], e)
    if e.name then
        self.item_byname[e.name] = e --name is unique
    end
    if e.loc then
        self.map:add_item(e) --add it to the MazeMap
    end
    return e
end
---------end add items----------------------

---------remove items----------------------
function MazeBase:remove(item, l)
--remove item from table l
	for i = 1, #l do
		if l[i]==item then --by reference
			table.remove(l,i) --remove the reference
		end
	end
end

function MazeBase:remove_item(item)
	self:remove(item, self.items)
	if item.type then
		self:remove(item,self.items_bytype[item.type]) --remove the reference
	end
	if item.name then
		self.items_byname[item.name] = nil
	end
	if item.loc then
		self.map:remove_item(item)
	end
end

--original code from CommNet
function MazeBase:remove_byloc(y, x, t)
    local l = self.map.items[y][x]
    local r = {}
    for i = 1, #l do
        if l[i].type == t then
            table.insert(r, l[i])
        end
    end
    for i = 1, #r do
        self:remove_item(r[i])
    end
end

--original code from CommNet
function MazeBase:remove_bytype(type)
    local l = self.items_bytype[type]
    local r = {}
    for i = 1, #l do
        table.insert(r, l[i])
    end
    for i = 1, #r do
        self:remove_item(r[i])
    end
end

--original code from CommNet
function MazeBase:remove_byname(name)
    self:remove_item(self.item_byname[name])
end
---------end remove items----------------------


function MazeBase:act(action_id)
	--self.agent: the current active agent
    self.agent:act(action_id)
end

