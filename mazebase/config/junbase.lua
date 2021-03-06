if not g_opts then g_opts = {} end
g_opts.multigames = {}
-------------------
--some shared RangeOpts
--current min, current max, min max, max max, increment
local mapH = torch.Tensor{6,6,5,10,1}
local mapW = torch.Tensor{6,6,5,10,1}
local blockspct = torch.Tensor{.0,.0, 0,.2,.01}
local waterpct = torch.Tensor{.0,.0, 0,.2,.01}

-------------------
--some shared StaticOpts
local sso = {}
-------------- costs:
sso.costs = {}
sso.costs.goal = -1
sso.costs.empty = 0.1
sso.costs.block = 1000
sso.costs.water = 0.3
sso.costs.corner = 0
sso.costs.step = 0.1
sso.costs.pushableblock = 1000
---------------------
sso.toggle_action = 0
sso.crumb_action = 0
sso.push_action = 0
sso.flag_visited = 0
sso.enable_corners = 0
sso.enable_boundary = 1
sso.max_attributes = g_opts.max_attributes or 6
-------------------------
sso.num_types_objects = 3
sso.min_num_objects = 1 --min number of a type of object
sso.max_num_objects = 1 --max number of a type of object
-------------------------------------------------------
sso.num_symbols = 5
sso.listener_visibility = 0
sso.pickup_enable =false

-- g_opts----
g_opts.map_height = mapH[1]
g_opts.map_width = mapW[1]
g_opts.num_symbols = sso.num_symbols
g_opts.num_types_objects = sso.num_types_objects
g_opts.listener_visibility = sso.listener_visibility
g_opts.pickup_enable = sso.pickup_enable
if sso.pickup_enable == false then
	g_opts.listener_nactions = 5     --moves 
	g_opts.num_tasks = sso.num_types_objects     --moves 
else
	g_opts.listener_nactions = 5 + 4     --moves + pickup
	g_opts.num_tasks = sso.num_types_objects*2     --moves + pickup
end



-- JunBase:
local JunBaseRangeOpts = {}
JunBaseRangeOpts.mapH = mapH:clone()
JunBaseRangeOpts.mapW = mapW:clone()
JunBaseRangeOpts.blockspct = blockspct:clone()
JunBaseRangeOpts.waterpct = waterpct:clone()

local JunBaseStaticOpts = {}
for i,j in pairs(sso) do JunBaseStaticOpts[i] = j end

JunBaseOpts ={}
JunBaseOpts.RangeOpts = JunBaseRangeOpts
JunBaseOpts.StaticOpts = JunBaseStaticOpts

g_opts.multigames.JunBase = JunBaseOpts


return g_opts
