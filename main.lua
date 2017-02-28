

torch.setdefaulttensortype('torch.FloatTensor')
paths.dofile('games/init.lua')

local cmd = torch.CmdLine()

--game parameters
cmd:option('--nagents', 1, 'the number of agents')
cmd:option('--nactions', 6, 'the number of agent actions')
cmd:option('--max_steps', 20, 'force to end the game after this many steps')
cmd:option('--games_config_path', 'games/config/crossing.lua', 'configuration file for games')
cmd:option('--game', '', 'can specify a single game')
cmd:option('--visibility', 1, 'vision range of agents')

g_opts = cmd:parse(arg or {})

g_init_game() --create g_factory
g_init_vocab() --create g_vocab
print(g_opts.game) 