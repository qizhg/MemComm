-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

if not g_opts then g_opts = {} end

g_opts.multigames = {}
g_opts.max_attributes = 5 --?

local mapH = torch.Tensor{16,16,16,16,1}
local mapW = torch.Tensor{16,16,16,16,1}

-------------------
--some shared StaticOpts
local sso = {}
-------------- costs:
sso.costs = {}
sso.costs.goal = 0
sso.costs.step = 0
sso.costs.pass = -10
sso.costs.collision = 0.1
sso.costs.wait = 0.01
---------------------
sso.visibility = g_opts.visibility
sso.nagents = g_opts.nagents
sso.max_agents = g_opts.nagents
sso.max_attributes = g_opts.max_attributes

-------------------------------------------------------
local CrossingRangeOpts = {}
CrossingRangeOpts.mapH = mapH:clone()
CrossingRangeOpts.mapW = mapW:clone()
CrossingRangeOpts.add_rate = torch.Tensor{0.05,0.05,0.05,0.2,0.01}


local CrossingStaticOpts = {}
for i,j in pairs(sso) do CrossingStaticOpts[i] = j end

local CrossingOpts ={}
CrossingOpts.RangeOpts = CrossingRangeOpts
CrossingOpts.StaticOpts = CrossingStaticOpts

g_opts.multigames.Crossing = CrossingOpts

return g_opts
