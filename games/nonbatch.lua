

function nonbatch_obs(game)
    local input = torch.Tensor(2*g_opts.visibility+1, 2*g_opts.visibility+1, g_opts.nwords)
    input:fill(game.vocab['nil'])
    game:get_visible_state(input)
    input = input:view(-1)
    return input
end
