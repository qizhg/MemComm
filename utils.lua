function get_agent(g, a)
    if torch.type(g) == 'CombatGame' then
        local c = 0
        for i = 1, g.nagents do
            if (not g.agents[i].team) or g.agents[i].team == 'team1' then
                c = c + 1
                if c == a then
                    return g.agents[i]
                end
            end
        end
        error('can not find agent ' .. a)
    else
        return g.agents[a]
    end
end

function set_current_agent(g, a)
    g.agent = get_agent(g, a)
end