
--torch.manualSeed(45)
torch.setdefaulttensortype('torch.FloatTensor')
paths.dofile('util.lua')
paths.dofile('model.lua')
paths.dofile('train.lua')
paths.dofile('games/init.lua')

local cmd = torch.CmdLine()
-- model parameters
cmd:option('--model', 'mlp', 'module type: mlp | rnn | lstm')
cmd:option('--nhop', 1, 'the number of model steps per action')
cmd:option('--hidsz', 10, 'the size of the internal state vector')
cmd:option('--memsize', 1, 'memorize the last 3 time steps')
cmd:option('--nonlin', 'relu', 'non-linearity type: tanh | relu | none')
cmd:option('--init_std', 0.2, 'STD of initial weights')
cmd:option('--init_hid', 0.1, 'initial value of internal state')
cmd:option('--unshare_hops', false, 'not share weights of different hops')
cmd:option('--encoder_lut', false, 'use LookupTable in encoder instead of Linear')
cmd:option('--encoder_lut_size', 50, 'max items in encoder LookupTable')
cmd:option('--unroll', 10, 'unroll steps for recurrent model. 0 means full unrolling.')
cmd:option('--unroll_freq', 4, 'unroll after every several steps')
-- game parameters
cmd:option('--nagents', 1, 'the number of agents')
cmd:option('--nactions', 5, 'the number of agent actions')
cmd:option('--max_steps', 30, 'force to end the game after this many steps')
cmd:option('--games_config_path', 'games/config/crossing.lua', 'configuration file for games')
cmd:option('--game', '', 'can specify a single game')
cmd:option('--visibility', 1, 'vision range of agents')
-- training parameters
cmd:option('--optim', 'rmsprop', 'optimization method: rmsprop | sgd | adam')
cmd:option('--lrate', 3e-3, 'learning rate')
cmd:option('--max_grad_norm', 0, 'gradient clip value')
cmd:option('--clip_grad', 0, 'gradient clip value')
cmd:option('--alpha', 0.03, 'coefficient of baseline term in the cost function')
cmd:option('--epochs', 50, 'the number of training epochs')
cmd:option('--nbatches', 100, 'the number of mini-batches in one epoch')
cmd:option('--batch_size', 4, 'size of mini-batch (the number of parallel games) in each thread')
cmd:option('--nworker', 1, 'the number of threads used for training')
cmd:option('--reward_mult', 1, 'coeff to multiply reward for bprop')
-- for optim
cmd:option('--momentum', 0, 'momentum for SGD')
cmd:option('--wdecay', 0, 'weight decay for SGD')
cmd:option('--rmsprop_alpha', 0.97, 'parameter of RMSProp')
cmd:option('--rmsprop_eps', 1e-6, 'parameter of RMSProp')
cmd:option('--adam_beta1', 0.9, 'parameter of Adam')
cmd:option('--adam_beta2', 0.999, 'parameter of Adam')
cmd:option('--adam_eps', 1e-8, 'parameter of Adam')
-- continuous communication with CommNet
cmd:option('--comm', false, 'enable continuous communication (CommNet)')
cmd:option('--comm_mode', 'avg', 'operation on incoming communication: avg | sum')
cmd:option('--comm_scale_div', 1, 'divide comm vectors by this')
cmd:option('--comm_encoder', 0, 'encode incoming comm: 0=identity | 1=linear')
cmd:option('--comm_decoder', 1, 'decode outgoing comm: 0=identity | 1=linear | 2=nonlin')
cmd:option('--comm_zero_init', false, 'initialize comm weights to zero')
cmd:option('--comm_range', 0, 'disable comm if L0 distance is greater than range')
-- discrete communication and other baselines
cmd:option('--nactions_comm', 1, 'enable discrete communication when larger than 1')
cmd:option('--dcomm_entropy_cost', 0, 'entropy regularization for discrete communication')
cmd:option('--fully_connected', false, 'use fully-connected model for all agents')
--other
cmd:option('--save', '', 'file name to save the model')
cmd:option('--load', '', 'file name to load the model')
cmd:option('--show', false, 'show progress')
cmd:option('--no_coop', false, 'agents are NOT cooperative')
cmd:option('--plot', false, 'plot average reward during training')
cmd:option('--curriculum_sta', 0, 'start making harder after this many epochs')
cmd:option('--curriculum_end', 0, 'when to make the game hardest')

g_opts = cmd:parse(arg or {})
print(g_opts)

g_init_game() --create g_factory
g_init_vocab() --create g_vocab
g_factory.vocab = g_vocab

print(g_factory.vocab)

g_init_model()
g_log = {}
train(g_opts.epochs)
