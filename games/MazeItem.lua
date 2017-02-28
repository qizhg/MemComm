local MazeItem = torch.class('MazeItem')

--An item can have attributes such as:
----type: such as "water", "block"
----loc: absolute coordinate
----name: unique name (optional)

function MazeItem:__init(attr)
	self.type = attr.type
	self.name = attr.name
	self.attr = attr
	self.loc  = self.attr.loc  
end

function MazeItem:is_reachable()
	if self.type == 'block' then return false end

	if self.type == 'door' then
		if self.attr.open == 'open' then
			return true
		else
			return false
		end
	end

	return true
end


--original code from CommNet
function MazeItem:to_sentence(dy, dx, disable_loc)
    local s = {}
    for k,v in pairs(self.attr) do
        if k == 'loc' then
            if not disable_loc then
                local y = self.loc.y - dy
                local x = self.loc.x - dx
                table.insert(s, 'y' .. y .. 'x' .. x)
            end
            if self.abs_loc_visible then
                table.insert(s, 'ay' .. self.loc.y .. 'x' .. self.loc.x)
            end
        elseif type(k) == 'string' and k:sub(1,1) == '_' then
            -- skip
        else
            table.insert(s, v)
        end
    end
    return s
end