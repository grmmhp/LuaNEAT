# LuaNEAT

LuaNEAT is a Lua implementation of the NEAT algorithm. LuaNEAT was designed with Love2D in mind but works in vanilla Lua.

# What is NEAT?
NEAT (NeuroEvolution of Augmenting Topologies) is an algorithm developed by Kenneth Stanley and Risto Miikkulainen for evolving neural networks topologies along with the weights of its connections. This implementation of NEAT is based on sethbling's MarI/O and the one from the book 'AI Techniques for Game Programming' by Mat Buckland. Read the original paper for more information on NEAT: https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

# Getting started

Simply put the LuaNEAT.lua file into your project's folder!

# Basic usage

``` lua
local LuaNEAT = require"LuaNEAT"

---------------------
--- initalization ---
---------------------

size = 150             -- the number of neural networks to be tested at each generation
num_inputs = 3         -- number of inputs of the neural nets
num_outputs = 2        -- number of outputs of the neural nets
force_no_bias = true   -- whether the neural nets will have a bias neuron or not

-- creating the pool:
-- keep in mind that LuaNEAT generates neural nets with a bias neuron by default (an extra input neuron whose value is always 1)
-- you can disable it by ignoring the bias argumenting when creating the pool
pool = LuaNEAT.newPool(size, num_inputs, num_outputs, force_no_bias)

-- initializing the pool:
pool:initialize()

-- getting the neural networks from the pool:
neural_nets = pool:getNeuralNetworks()

---------------------
---   evaluation  ---
---------------------

local max_generation = 10000
local fitness_goal = 100

-- running the algorithm
while true do
  -- evaluating all the neural nets
  for _,network in ipairs(neural_nets) do
    -- assign a fitness number to each network in the pool
    network:setFitness(fit)
  end

  -- all networks have been evaluated;
  -- next generation of networks can be computed:
  pool:nextGeneration()

  -- we're stopping when a network reaches the desired fitness goal
  -- or the maximum number of generations has been reached
  if pool:getGeneration() == max_generation or pool:getLastBestFitness() >= fitness_goal then
    break
  end
end
```
