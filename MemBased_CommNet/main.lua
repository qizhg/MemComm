
torch.setdefaulttensortype('torch.FloatTensor')
paths.dofile('util.lua')
paths.dofile('model.lua')
--paths.dofile('train.lua')
paths.dofile('games/init.lua')

require'gnuplot'

local cmd = torch.CmdLine()
-- model parameters
cmd:option('--nhop', 1, 'the number of model steps per action')
cmd:option('--hidsz', 20, 'the size of the internal state vector')
cmd:option('--memsize', 10, 'memorize the last 3 time steps')
cmd:option('--nonlin', 'tanh', 'non-linearity type: tanh | relu | none')
cmd:option('--init_std', 0.2, 'STD of initial weights')
-- game parameters
cmd:option('--nagents', 10, 'the number of agents')
cmd:option('--nactions', 2, 'the number of agent actions')
cmd:option('--max_steps', 40, 'force to end the game after this many steps')
cmd:option('--games_config_path', 'games/config/crossing.lua', 'configuration file for games')
cmd:option('--visibility', 1, 'vision range of agents')
-- comm
cmd:option('--comm', true, 'enable continuous communication (CommNet)')
--cmd:option('--comm_mode', 'avg', 'operation on incoming communication: avg | sum')
-- training parameters
cmd:option('--optim', 'rmsprop', 'optimization method: rmsprop | sgd | adam')
cmd:option('--lrate', 3e-3, 'learning rate')
cmd:option('--alpha', 0.03, 'coefficient of baseline term in the cost function')
cmd:option('--beta', 0, 'coefficient of baseline term in the cost function')
cmd:option('--eps_start', 0.2, 'eps')
cmd:option('--eps_end', 0.05, 'eps')
cmd:option('--eps_end_batch', 10000, 'eps')
cmd:option('--epochs', 100, 'the number of training epochs')
cmd:option('--nbatches',200, 'the number of mini-batches in one epoch')
cmd:option('--batch_size', 5, 'size of mini-batch (the number of parallel games) in each thread')
cmd:option('--nworker', 1, 'the number of threads used for training')
cmd:option('--reward_mult', 1, 'coeff to multiply reward for bprop')
cmd:option('--max_grad_norm', 0, 'gradient clip value')
cmd:option('--clip_grad', 0, 'gradient clip value')
-- for optim
cmd:option('--momentum', 0, 'momentum for SGD')
cmd:option('--wdecay', 0, 'weight decay for SGD')
cmd:option('--rmsprop_alpha', 0.97, 'parameter of RMSProp')
cmd:option('--rmsprop_eps', 1e-6, 'parameter of RMSProp')
cmd:option('--adam_beta1', 0.9, 'parameter of Adam')
cmd:option('--adam_beta2', 0.999, 'parameter of Adam')
cmd:option('--adam_eps', 1e-8, 'parameter of Adam')
--other
cmd:option('--save', '', 'file name to save the model')
cmd:option('--load', 'mem10at70', 'file name to load the model')

g_opts = cmd:parse(arg or {})

g_init_game() --create g_factory
g_init_vocab() --create g_vocab
g_factory.vocab = g_vocab


g_init_model()
