# LuaNEAT

LuaNEAT is a Lua implementation of the NeuroEvolution of Augmenting Topologies algorithm

# What is NEAT?

Developed by Kenneth Stanley and Risto Miikkulainen, NEAT is an algorithm for evolving
neural networks topologies along with their weights.

# Getting started

Simply put the LuaNEAT.lua file into your project's folder!

# Basic usage

``` lua
local LuaNEAT = require"LuaNEAT"

---------------------
--- initalization ---
---------------------

size = 150         -- the number of neural networks to be tested at each generation
num_inputs = 3     -- number of inputs of the neural nets
num_outputs = 2    -- number of outputs of the neural nets
bias = true        -- whether the neural nets will have a bias neuron or not

-- creating the pool:
-- keep in mind that LuaNEAT generates neural nets with a bias neuron by default (an extra input neuron whose value is always 1)
-- you can disable it by ignoring the bias argumenting when creating the pool
pool = LuaNEAT.newPool(size, num_inputs, num_outputs, bias)

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
