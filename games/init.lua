-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

paths.dofile('MazeBase.lua')
paths.dofile('OptsHelper.lua')
paths.dofile('GameFactory.lua')
paths.dofile('batch.lua')

-- for traffic games
paths.dofile('Traffic.lua')
paths.dofile('Crossing.lua')
--paths.dofile('CrossingEasy.lua')
--paths.dofile('CrossingHard.lua')

-- for combat games
--paths.dofile('CombatAgent.lua')
--paths.dofile('CombatAgentFactory.lua')
--paths.dofile('CombatGame.lua')
--paths.dofile('CombatGameFactory.lua')


local function init_game_opts()
    if g_opts.team_control then
        -- combat game
        g_factory = CombatGameFactory(g_opts,g_vocab)
        return
    end

    local games = {}
    local helpers = {}
    games.Crossing = Crossing
    helpers.Crossing = OptsHelper
    games.CrossingEasy = CrossingEasy
    helpers.CrossingEasy = OptsHelper
    games.CrossingHard = CrossingHard
    helpers.CrossingHard = OptsHelper
    games.CombatGame = CombatGame
    helpers.CombatGame = CombatGameFactory

    g_factory = GameFactory(g_opts,g_vocab,games,helpers)

    return games, helpers
end

function g_init_vocab()
    local function vocab_add(word)
        if g_vocab[word] == nil then
            local ind = g_opts.nwords + 1
            g_opts.nwords = g_opts.nwords + 1
            g_vocab[word] = ind
            g_ivocab[ind] = word
        end
    end
    g_vocab = {}
    g_ivocab = {}
    g_ivocabx = {}
    g_ivocaby = {}
    g_opts.nwords = 0

    -- general
    vocab_add('nil')
    vocab_add('agent')
    vocab_add('block')

    -- absolute coordinates
    --for y = 1, 8 do
    --    for x = 1, 8 do
    --        vocab_add('ay' .. y .. 'x' .. x)
    --    end
    --end

    --route
    for i = 1, 3 do
        vocab_add('route' .. i)
    end

end

function g_init_game()
    g_opts = dofile(g_opts.games_config_path)
    local games, helpers = init_game_opts()
end

function new_game()
    if g_opts.game == nil or g_opts.game == '' then
        return g_factory:init_random_game()
    else
       return g_factory:init_game(g_opts.game)
    end
end
