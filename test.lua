
--torch.manualSeed(451)

local function init()
    require('xlua')
	torch.setdefaulttensortype('torch.FloatTensor')
	paths.dofile('util.lua')
	--paths.dofile('model/listener_model.lua')
	paths.dofile('model/speaker_model.lua')
	paths.dofile('train.lua')
    --paths.dofile('listener_train.lua')
    paths.dofile('speaker_train.lua')
	paths.dofile('mazebase/init.lua')
end

local function init_threads()
    print('starting ' .. g_opts.nworker .. ' workers')
    local threads = require('threads')
    threads.Threads.serialization('threads.sharedserialize')
    local workers = threads.Threads(g_opts.nworker, init)
    workers:specific(true)
    for w = 1, g_opts.nworker do
        workers:addjob(w,
            function(opts_orig, vocab_orig)
                g_opts = opts_orig
                g_vocab = vocab_orig
                g_init_listener_model()
				g_init_speaker_model()
                g_mazebase.init_game()
            end,
            function() end,
            g_opts, g_vocab
        )
    end
    workers:synchronize()
    return workers
end

init()


local cmd = torch.CmdLine()
-- threads
cmd:option('--nworker', 1, 'the number of threads used for training')
-- model parameters
cmd:option('--hidsz', 20, 'the size of the internal state vector')
cmd:option('--nonlin', 'relu', 'non-linearity type: tanh | relu | none')
cmd:option('--init_std', 0.1, 'STD of initial weights')
cmd:option('--lstm', true, '')
-- game parameters
cmd:option('--max_steps', 20, 'force to end the game after this many steps')
cmd:option('--games_config_path', 'mazebase/config/junbase.lua', 'configuration file for games')
-- training parameters
---------
cmd:option('--epochs', 100, 'the number of training epochs')
cmd:option('--nbatches', 100, 'the number of mini-batches in one epoch')
cmd:option('--batch_size',512, 'size of mini-batch (the number of parallel games) in each thread')
---- GAE
cmd:option('--gamma', 0.99, 'size of mini-batch (the number of parallel games) in each thread')
cmd:option('--lambda', 0.95, 'size of mini-batch (the number of parallel games) in each thread')
---- lr
cmd:option('--lrate', 1e-3, 'learning rate')
---- Gumbel
cmd:option('--Gumbel_temp', 1.0, 'fixed Gumbel_temp')
---- baseline mixing
cmd:option('--alpha', 0.03, 'coefficient of baseline term in the cost function')
---- entropy mixing
cmd:option('--beta_start', 0.01, 'coefficient of listener entropy mixing')
cmd:option('--beta_end_batch', 100*70, '')
---- clipping
cmd:option('--reward_mult', 1, 'coeff to multiply reward for bprop')
cmd:option('--max_grad_norm', 0, 'gradient clip value')
cmd:option('--clip_grad', 0, 'gradient clip value')
-- for optim
cmd:option('--optim', 'rmsprop', 'optimization method: rmsprop | sgd | adam')
cmd:option('--momentum', 0, 'momentum for SGD')
cmd:option('--wdecay', 0, 'weight decay for SGD')
cmd:option('--rmsprop_alpha', 0.97, 'parameter of RMSProp')
cmd:option('--rmsprop_eps', 1e-6, 'parameter of RMSProp')
cmd:option('--adam_beta1', 0.9, 'parameter of Adam')
cmd:option('--adam_beta2', 0.999, 'parameter of Adam')
cmd:option('--adam_eps', 1e-8, 'parameter of Adam')
--other
cmd:option('--save', '', 'file name to save the model')
cmd:option('--load', '', 'file name to load the model')
g_opts = cmd:parse(arg or {})


g_mazebase.init_vocab()
g_mazebase.init_game()

if g_opts.nworker > 1 then
    g_workers = init_threads()
end

g_log = {}
g_init_speaker_model()
g_load_model()

--[[
train(g_opts.epochs - #g_log)
g_save_model()
--]]

speaker_train(g_opts.epochs - #g_log)



