-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

local CrossingEasy, parent = torch.class('CrossingEasyIndicator', 'Traffic')

function CrossingEasy:__init(opts, vocab)
    parent.__init(self, opts, vocab)
    self.name="ind"
end

function CrossingEasy:build_roads()
    -- build crossing
    assert(self.map.height % 2 == 1)
    assert(self.map.height == self.map.width)
    self.length = math.floor(self.map.height / 2)
    for y = 1, self.length do
        self:place_item({type = 'block'}, y, self.length)
        self:place_item({type = 'block'}, y, self.length + 2)
    end
    for y = self.length+2, self.map.height do
        self:place_item({type = 'block'}, y, self.length)
        self:place_item({type = 'block'}, y, self.length + 2)
    end
    for x = 1, self.length-1 do
        self:place_item({type = 'block'}, self.length, x)
        self:place_item({type = 'block'}, self.length + 2, x)
    end
    for x = self.length+3, self.map.width do
        self:place_item({type = 'block'}, self.length, x)
        self:place_item({type = 'block'}, self.length + 2, x)
    end

    self:place_item({type = 'block'}, self.length + 2, self.length + 1)
    self:place_item({type = 'block'}, self.length , self.length + 1)
    table.insert(self.source_locs, {y = self.length + 1, x = self.length + 1, routes = {}})

    local dst_id = torch.random(2)
    self:place_item({type = 'route'..dst_id}, self.length + 1, self.length + 1)
    if dst_id == 1 then 
        r = {}
        table.insert(r, {y = self.length + 1, x =self.map.width})
        table.insert(self.routes, r)
        table.insert(self.source_locs[1].routes, #self.routes)
    elseif  dst_id == 2 then 
        r = {}
        table.insert(r, {y = self.length + 1, x =1})
        table.insert(self.routes, r)
        table.insert(self.source_locs[1].routes, #self.routes)
    else  
        r = {}
        table.insert(r, {y = self.map.height, x = self.length + 1})
        table.insert(self.routes, r)
        table.insert(self.source_locs[1].routes, #self.routes)
    end
    --print(#self.routes)
   
end
