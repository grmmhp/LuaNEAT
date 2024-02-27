local LuaNEAT = {
  _VERSION = "LuaNEAT Beta 1.0",
  _DESCRIPTION = "NEAT module for Lua",
  _URL = "https://github.com/grmmhp/LuaNEAT",
  _LICENSE = [[
    MIT License

    Copyright (c) 2023 Gabriel Mesquita

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
  ]],
}

--math.randomseed(os.time())
--love.math.randomseed(os.time())
LuaNEAT.random = love.math.random or math.random

function gaussian(mean, variance) --https://rosettacode.org/wiki/Statistics/Normal_distribution#Lua
    return  math.sqrt(-2 * variance * math.log(LuaNEAT.random())) *
            math.cos(2 * math.pi * LuaNEAT.random()) + mean
end

--TODO:
-- check alter response
-- remove inability to link two output neurons (?)
-- remove species:adjustFitnesses()
-- write other activation functions
-- pass activation function to neural net

-- remove activation from NeuronGene
-- maybe(?) add "remove node" and "remove link" mutations


--------------------------------
--         PARAMETERS         --
--------------------------------

LuaNEAT.parameters = {
  -- general parameters
  defaultActivation = "sigmoid",
  defaultResponse = 4.9,
  findLinkAttempts = 20,
  hiddenNeuronsThreshold = 5,
  maxNeuronsAmount = 1e6,
  initialWeightRange = 2,

  -- speciation
  excessGenesCoefficient   = 2,
  disjointGenesCoefficient = 2,
  matchingGenesCoefficient = .4,
  sameSpeciesThreshold     = 1,

  maxStaleness             = 15,
  tournamentSize           = 3,

  -- mutation rates and parameters
  mutateMutationRates = true,
  mutation_rates = {
    weightStep          = .1,
    responseStep        = .1,
    addLink             = 2,
    addNode             = .3,
    loopedLink          = .1,
    enableDisable       = .1,

    perturbWeight   = .25,
    replaceWeight   = .1,
    maxPerturbation = .1,
    perturbResponse = 0,--.01,
    alterResponse   = 0,--.2,
    maxResponse     = 2,
  },

  -- crossover
  crossoverRate = .75,
}

local function copy_mutation_rates(parameters)
  local rates = {}
  local parameters = parameters or LuaNEAT.parameters
  for k,v in pairs(parameters.mutation_rates) do
    rates[k] = v
  end

  return rates
end

--[[local _DEBUG = false
local _DEBUG_TRADITIONAL_FEEDFORWARD_MODE = false
local _DEBUG_HIDDEN_LAYERS = {}

function LuaNEAT._DEBUG_SET_HIDDEN_LAYERS(layers)
  _DEBUG_HIDDEN_LAYERS = layers
end

if _DEBUG_TRADITIONAL_FEEDFORWARD_MODE then
  LuaNEAT.parameters.mutation_rates.addLink           = 0
  LuaNEAT.parameters.mutation_rates.addNode           = 0
  LuaNEAT.parameters.mutation_rates.loopedLink        = 0
  LuaNEAT.parameters.mutation_rates.enableDisable     = 0
end

local oldprint = print

if _DEBUG then
  print = function() return; end
end]]

--------------------------------
--    ACTIVATION FUNCTIONS    --
--------------------------------

local activations = {
  ["sigmoid"] = function(x,p)
    return 2/(1+math.exp(-x*p)) - 1
  end,

  ["positive-sigmoid"] = function(x,p)
    return 1/(1+math.exp(-x*p))
  end,

  ["sine"] = function(x,p)
    return math.sin(x*p)
  end,

  ["ReLU"] = function(x,p)
    return math.max(0, math.min(x*p, 1))
  end,
}


-- forward declaration of classes
local Genome, NeuronGene, LinkGene, NeuralNetwork, InnovationList, Species, Pool, Statistics

--------------------------------
--           GENOME           --
--------------------------------

Genome = {}
Genome.mt = {
  __index = Genome,
}

setmetatable(Genome, {
  __call = function(t, id, neuron_list, link_list, mutation_rates)
    local o = {}

    o.id = id or -1
    o.species = -1
    o.fitness = 0
    o.number_of_inputs = 0
    o.number_of_outputs = 0
    o.bias = 0
    o.neuron_list = neuron_list or {}
    o.link_list = link_list or {}
    o.mutation_rates = mutation_rates or {}

    return setmetatable(o, Genome.mt)
  end
})

function Genome.minimal(id, inputs, outputs, parameters, mutation_rates, noBias, disconnected)
  -- creates a minimal genome with a given number of inputs and outputs
  -- where every input neuron is connected to a output neuron
  -- it also records the number of inputs on the genome

  -- dx is used to position input, output and bias neurons evenly spaced on the grid
  local genome
  local neuron_list = {}
  local link_list = {}
  local dx = 1/(inputs+2)

  if noBias then
    dx = 1/(inputs+1)
  end

  for n=1,inputs do
    local gene = NeuronGene(n, "input", parameters.defaultActivation, false, parameters.defaultResponse, n*dx, 1)
    table.insert(neuron_list, gene) --id, neuron_type, activation, recurrent, response, x, y
  end

  if not noBias then
    local gene = NeuronGene(inputs+1, "bias", parameters.defaultActivation, false, parameters.defaultResponse, (inputs+1)*dx, 1)
    table.insert(neuron_list, gene)
    inputs = inputs + 1
  end

  dx = 1/(outputs+1)
  for n=1,outputs do
    local gene = NeuronGene(inputs+n, "output", parameters.defaultActivation, false, parameters.defaultResponse, n*dx, 0)
    table.insert(neuron_list, gene)

    -- creating links
    if not disconnected then
      for k=1,inputs do
        local innovation = k + (inputs)*(n-1)
        local link = LinkGene(innovation, 0, k, inputs+n, true, false)--innovation, weight, from, to, enabled, recurrent
        link:randomWeight(parameters.initialWeightRange)

        table.insert(link_list, link)
      end
    end
  end

  genome = Genome(id, neuron_list, link_list, mutation_rates)
  genome.number_of_inputs = inputs
  genome.number_of_outputs = outputs
  if not noBias then genome.bias = 1 end

  return genome
end

function Genome.layered(id, inputs, outputs, hidden_layers, parameters, mutation_rates, noBias, disconnected)
  -- creates a traditional feedforward neural network genome with a given number of inputs and outputs;
  -- hidden_layers is an array containg the number of neurons per hidden layer in a total of #hidden_layers layers
  -- where every neuron in a layer is connected to every neuron in the previous layer
  -- it also records the number of inputs on the genome

  -- dx is used to position input, output and bias neurons evenly spaced on the grid
  local genome
  local neuron_list = {}
  local link_list = {}
  local dx = 1/(inputs+2)
  local hidden_neurons_id_counter = inputs+outputs+2

  if noBias then
    dx = 1/(inputs+1)

    hidden_neurons_id_counter = hidden_neurons_id_counter - 1
  end

  for n=1,inputs do
    local gene = NeuronGene(n, "input", parameters.defaultActivation, false, parameters.defaultResponse, n*dx, 1)
    table.insert(neuron_list, gene) --id, neuron_type, activation, recurrent, response, x, y
  end

  if not noBias then
    local gene = NeuronGene(inputs+1, "bias", parameters.defaultActivation, false, parameters.defaultResponse, (inputs+1)*dx, 1)
    table.insert(neuron_list, gene)
    inputs = inputs + 1
  end

  dx = 1/(outputs+1)
  for n=1,outputs do
    local gene = NeuronGene(inputs+n, "output", parameters.defaultActivation, false, parameters.defaultResponse, n*dx, 0)
    table.insert(neuron_list, gene)
  end

  local layers = hidden_layers
  local function totalNumHiddenNeurons()
    local count = 0

    for n=1,#layers do
      count = count + layers[n]
    end

    return count
  end

  local function numNeuronsBeforeLayer(j)
    if j==1 then return inputs end

    local count = inputs

    for n=1, j-1 do
      count = count+layers[n]
    end

    return count
  end


  local dy = 1/(#layers+1)
  local innovation_counter = 1
  for l=1, #layers do
    for n=1, layers[l] do
      local dx = 1/(layers[l]+1)
      local gene = NeuronGene(hidden_neurons_id_counter, "hidden", parameters.defaultActivation, false, parameters.defaultResponse, n*dx, (#layers-l+1)*dy)
      table.insert(neuron_list, gene)

      if not disconnected then
        if l==1 then
          for j=1, inputs do
            local link =  LinkGene(innovation_counter, 0, j, hidden_neurons_id_counter, true, false)--innovation, weight, from, to, enabled, recurrent
            link:randomWeight(parameters.initialWeightRange)
            table.insert(link_list, link)

            innovation_counter = innovation_counter + 1
          end
        else
          for j=1, layers[l-1] do
            local num_so_far = numNeuronsBeforeLayer(l-1)
            --print("neuron ".. j .. " from layer ".. l .. " with a total of ".. layers[l].. "neurons; total num of neurons before this layer is ".. num_so_far)
            local link =  LinkGene(innovation_counter, 0, num_so_far+outputs+j, hidden_neurons_id_counter, true, false)--innovation, weight, from, to, enabled, recurrent
            link:randomWeight(parameters.initialWeightRange)
            table.insert(link_list, link)

            innovation_counter = innovation_counter + 1
          end
        end

        hidden_neurons_id_counter = hidden_neurons_id_counter + 1
      end
    end
  end


  if not disconnected then
    local num_so_far = numNeuronsBeforeLayer(#layers)
    for n=1,outputs do
      -- creating links
      for k=1,layers[#layers] do
        local link = LinkGene(innovation_counter, 0, num_so_far+outputs+k, inputs+n, true, false)--innovation, weight, from, to, enabled, recurrent
        link:randomWeight(parameters.initialWeightRange)

        table.insert(link_list, link)

        innovation_counter = innovation_counter + 1
      end
    end
  end



  genome = Genome(id, neuron_list, link_list, mutation_rates)
  genome.number_of_inputs = inputs
  genome.number_of_outputs = outputs
  if not noBias then genome.bias = 1 end

  return genome
end

function Genome.sameSpecies(genome1, genome2, parameters)
  local matching    = 0
  local excess      = 0
  local disjoint    = 0
  local difference  = 0 -- sum of abs val of weight difference between matching links

  local index1 = 1 -- genome 1 index
  local index2 = 1 -- genome 2 index

  while index1 <= #genome1.link_list and index2 <= #genome2.link_list do
    local link1 = genome1.link_list[index1]
    local link2 = genome2.link_list[index2]

    if link1.innovation == link2.innovation then
      matching = matching + 1
      difference = difference + math.abs(link1.weight - link2.weight)

      index1 = index1 + 1
      index2 = index2 + 1
    elseif link1.innovation < link2.innovation then
      disjoint = disjoint + 1
      index1 = index1 + 1
    elseif link2.innovation < link1.innovation then
      disjoint = disjoint + 1
      index2 = index2 + 1
    end
  end

  if index1 <= #genome1.link_list then
    excess = #genome1.link_list-index1 + 1
  end

  if index2 <= #genome2.link_list then
    excess = #genome2.link_list-index2 + 1
  end

  local maxIndex = math.max(#genome1.link_list, #genome2.link_list)

  local distance = (parameters.excessGenesCoefficient*excess
                  + parameters.disjointGenesCoefficient*disjoint)/maxIndex
                  + parameters.matchingGenesCoefficient*difference/matching;

  return distance < parameters.sameSpeciesThreshold
end

function Genome.crossover(genome1, genome2)
  local offspring = Genome()

  -- make sure genome1 is always the fittest
  -- if they are of equal fitness, choose genome1
  -- as the shorter genome. if both genomes have
  -- same length, just choose genome1 at random.

  if genome2.fitness > genome1.fitness then
    genome1, genome2 = genome2, genome1
  elseif genome2.fitness == genome1.fitness then
    if #genome1.link_list == #genome2.link_list then
      if LuaNEAT.random() < .5 then
        genome1, genome2 = genome2, genome1
      end
    elseif #genome2.link_list < #genome1.link_list then
      genome1, genome2 = genome2, genome1
    end
  end

  -- get the neuron genes from genome1
  for n=1,#genome1.neuron_list do
    local neuron = genome1.neuron_list[n]
    offspring:insertNeuron(neuron:copy(), true)
  end

  local iter1 = 1
  local iter2 = 1

  while iter1 <= #genome1.link_list and iter2 <= #genome2.link_list do
    local link1 = genome1.link_list[iter1]
    local link2 = genome2.link_list[iter2]

    if link1.innovation == link2.innovation then
      if LuaNEAT.random() < .5 then
        -- inheriting from genome1
        offspring:insertLink(link1:copy(), true)
      else
        -- inheriting from genome2
        local gene = (link2:copy())
        gene.from = link1.from      -- this prevents neuron id mismatch from ocurring (that can happen when resetting the innovation list every generation)
        gene.to = link1.to
        offspring:insertLink(gene, true)
      end
      iter1 = iter1 + 1
      iter2 = iter2 + 1
    elseif link1.innovation < link2.innovation then
      offspring:insertLink(link1:copy(), true)

      iter1 = iter1 + 1
    else
      iter2 = iter2 + 1
    end
  end

  for n = iter1, #genome1.link_list do
    local link = genome1.link_list[n]
    offspring:insertLink(link:copy(), true)
  end

  offspring.number_of_inputs = genome1.number_of_inputs
  offspring.number_of_outputs = genome1.number_of_outputs
  offspring.bias = genome1.bias

  -- copy genome 1's mutation rates
  for m,rate in pairs(genome1.mutation_rates) do
    offspring.mutation_rates[m]=rate
  end

  return offspring
end

function Genome:newNode(parameters, innovation_list)
  if #self.neuron_list >= parameters.maxNeuronsAmount then return end
  if #self.link_list == 0 then return end

  local neuron  -- the neuron to be added
  local from    -- the neuron with link going to the new neuron
  local to      -- the neuron with link coming to the new neuron
  local link    -- the random selected link to be split
  local threshold =   self.number_of_inputs
                    + self.number_of_outputs
                    + parameters.hiddenNeuronsThreshold

  if #self.link_list < threshold then
    -- genome size is too small for
    -- any link to be selected at random
    for _=1,parameters.findLinkAttempts do
      link = self:getRandomLink(true)
      from = self:getNeuron(link.from)

      -- make sure the link is enabled,
      -- is not recurrent and is not
      -- connected to a bias neuron

      if (link.enabled) and (not link.recurrent) and (from.neuron_type ~= "bias") then
        break
      else
        link = nil
      end
    end
  else
    -- genome is large enough for any
    -- link to be selected at random
    for n=1,#self.link_list do
      link = self:getRandomLink()
      from = self:getNeuron(link.from)

      -- make sure the link is enabled,
      -- is not recurrent and is not
      -- connected to a bias neuron

      if (link.enabled) and (not link.recurrent) and (from.neuron_type ~= "bias") then
        break
      end
    end
  end

  if not link then
    return "failed to add a neuron!"
  end

  to = self:getNeuron(link.to)

  local new_x = (from.x+to.x)/2
  local new_y = (from.y+to.y)/2

  local id = innovation_list:getNeuronID(from.id, to.id)

  -- "it is possible for NEAT to repeatedly do the following:
  -- 1. Find a link. Lets say we choose link 1 to 5
  -- 2. Disable the link,
  -- 3. Add a new neuron and two new links
  -- 4. The link disabled in Step 2 may be re-enabled when this genome
  -- is recombined with a genome that has that link enabled.
  -- 5 etc etc"

  -- "Therefore, the following checks to see if a neuron ID is already being used.
  -- If it is, the function creates a new innovation for the neuron"

  if id then
    if self:alreadyHaveThisNeuron(id) then
      id = nil
    end
  end

  if not id then
    -- this is a new innovation
    id = innovation_list:newNeuron(from.id, to.id)
    --id, neuron_type, activation, recurrent, response, x, y
    neuron = NeuronGene(
      id,
      "hidden",
      parameters.defaultActivation,
      false,
      parameters.defaultResponse,
      new_x,
      new_y
    )

    local incoming_id = innovation_list:newLink(from.id, id)
    local outgoing_id = innovation_list:newLink(id, to.id)

    --innovation, weight, from, to, enabled, recurrent
    local link_incoming = LinkGene(incoming_id, 1.0, from.id, id, true, false)
    local link_outgoing = LinkGene(outgoing_id, link.weight, id, to.id, true, false)

    self:insertNeuron(neuron, true)
    self:insertLink(link_incoming, true)
    self:insertLink(link_outgoing, true)
  else
    -- existing innovation
    local incoming_id = innovation_list:getLinkInnovation(from.id, id)
    local outgoing_id = innovation_list:getLinkInnovation(id, to.id)

    if (not incoming_id) or (not outgoing_id) then
      error("missing link innovation!")
    end

    --id, neuron_type, activation, recurrent, response, x, y
    local neuron = NeuronGene(
      id,
      "hidden",
      parameters.defaultActivation,
      false,
      parameters.defaultResponse,
      new_x,
      new_y
    )

    local link_incoming = LinkGene(incoming_id, 1.0, from.id, id, true, false)
    local link_outgoing = LinkGene(outgoing_id, link.weight, id, to.id, true, false)

    self:insertNeuron(neuron)
    self:insertLink(link_incoming)
    self:insertLink(link_outgoing)
  end

  link.enabled = false

  return "successfully added new node (".. id ..") between nodes ".. from.id .. " and ".. to.id
end

function Genome:newLink(parameters, mutation_rates, innovation_list, noLoop)
  local from, to
  local recurrent = false

  if (LuaNEAT.random() < mutation_rates.loopedLink) and (not noLoop) then
    -- a looped link will be created
    -- selects a hidden neuron to be selected for a looped link
    print("looped link to be added! chance is of ".. mutation_rates.loopedLink)

    for _=1,parameters.findLinkAttempts do
      local neuron = self:getRandomNeuron(true)

      -- checking if neuron already have a loop
      -- and if it isnt an input or bias neuron

      if  (not neuron.recurrent)
      and (neuron.neuron_type ~= "input")
      and (neuron.neuron_type ~= "bias") then
        neuron.recurrent = true
        recurrent = true
        from = neuron
        to = neuron

        break
      end
    end
  else
    -- a normal link will be created
    -- two random neurons will be selected
    -- the second one must not be input or bias
    -- and they must not have already been linked

    for _=1,parameters.findLinkAttempts do
      neuron1 = self:getRandomNeuron()
      neuron2 = self:getRandomNeuron(true)

      if  (neuron1.id ~= neuron2.id)
      and not (neuron1.neuron_type=="output" and neuron2.neuron_type=="output") --------------------------------------------------------------------------------------------------------------REMOOOOOOOOOOOOOOOOOVE
      and (not self:linkExists(neuron1.id, neuron2.id)) then
        from = neuron1
        to = neuron2

        break
      end
    end
  end

  -- failed to find link
  if not from then return "failed to add link!" end

  -- this new links innovation
  local id = innovation_list:getLinkInnovation(from.id, to.id)

  if not id then
    -- this is a new innovation
    id = innovation_list:newLink(from.id, to.id)
  end

  -- checking if link is recurrent
  -- outputs have y=0 and inputs y=1
  if from.y < to.y then recurrent=true; return end----------------------------------------------------------------------------------------------------------REMOVE THE RETURN

  -- innovation, weight, from, to, enabled, recurrent
  local gene = LinkGene(id, 0, from.id, to.id, true, recurrent)
  gene:randomWeight(parameters.initialWeightRange)

  self:insertLink(gene)
  return "successfully created new link between nodes ".. from.id .. " and ".. to.id
end

function Genome:mutate(parameters, innovation_list)
  if parameters.mutateMutationRates then
    for mut,rate in pairs(self.mutation_rates) do
      if LuaNEAT.random() < .5 then
        self.mutation_rates[mut] = 0.95*rate
      else
        self.mutation_rates[mut] = 1.05263*rate
      end
    end
  end

  -- mutating weights
  for n=1,#self.link_list do
    local link = self.link_list[n]
    if LuaNEAT.random() < self.mutation_rates.perturbWeight then
      if LuaNEAT.random() < self.mutation_rates.replaceWeight then
        link:randomWeight(parameters.initialWeightRange)
      else
        link.weight = link.weight + (LuaNEAT.random()*2-1)*self.mutation_rates.weightStep
      end
    end
  end

  -- perturbing neuron responses
  for n=1,#self.neuron_list do
    local neuron = self.neuron_list[n]
    if LuaNEAT.random() < self.mutation_rates.perturbResponse then
      if LuaNEAT.random() < self.mutation_rates.alterResponse then
        neuron:alterResponse(self.mutation_rates.maxResponse)
      else
        neuron.response = neuron.response + (LuaNEAT.random()*2-1)*self.mutation_rates.responseStep
      end
    end
  end

  -- randomly enabling or disabling links
  for n=1, #self.link_list do
    local link = self.link_list[n]
    if LuaNEAT.random() < self.mutation_rates.enableDisable then
      link.enabled = not link.enabled
    end
  end

  local p = self.mutation_rates.addNode
  while p>0 do
    if LuaNEAT.random() < p then
      self:newNode(parameters, innovation_list)
    end
    p = p - 1
  end

  p = self.mutation_rates.addLink
  while p>0 do
    if LuaNEAT.random() < self.mutation_rates.addLink then
      self:newLink(parameters, self.mutation_rates, innovation_list)
    end
    p = p - 1
  end
end

function Genome:linkExists(from, to)
  for n=1,#self.link_list do
    local link = self.link_list[n]

    if link.from == from and link.to == to then
      return true
    end
  end; return false
end

function Genome:getNeuron(id)
  --[[local lower, upper = 1, #self.neuron_list
  local index

  while lower <= upper do
    index = math.floor((lower+upper)/2)

    if self.neuron_list[index].id > id then
      upper = index-1
    elseif self.neuron_list[index].id < id then
      lower = index+1
    else
      return self.neuron_list[index], index
    end
  end]]

  for index=1,#self.neuron_list do
    if self.neuron_list[index].id == id then
      return self.neuron_list[index], index
    end
  end
end

function Genome:alreadyHaveThisNeuron(id)
  return self:getNeuron(id) ~= nil
end

function Genome:getRandomNeuron(noInputs)
  local lower = 1
  if noInputs then lower = self.number_of_inputs+1 end

  local index = LuaNEAT.random(lower, #self.neuron_list)
  return self.neuron_list[index]
end

function Genome:getRandomLink(oldLinkBias)
  local upper = #self.link_list
  if oldLinkBias then upper = upper-math.floor(math.sqrt(#self.link_list)) end

  local index = LuaNEAT.random(1, upper)
  return self.link_list[index]
end

function Genome:insertNeuron(gene, new)
  if new or #self.neuron_list == 0 then
    table.insert(self.neuron_list, gene); return
  end

  local lower, upper = 1, #self.neuron_list
  local index=0

  while lower <= upper do
    index = math.floor((lower+upper)/2)

    if self.neuron_list[index].id == gene.id then
      -- neuron is already on the list
      return
    elseif self.neuron_list[index].id > gene.id then
      upper = index-1
    elseif self.neuron_list[index].id < gene.id then
      lower = index+1
    end
  end

  if self.neuron_list[index].id < gene.id then
    table.insert(self.neuron_list, index+1, gene)
  else
    table.insert(self.neuron_list, index, gene)
  end
end

function Genome:insertLink(gene, new)
  if new or #self.link_list == 0 then
    table.insert(self.link_list, gene); return
  end

  local lower, upper = 1, #self.link_list
  local index=0

  while lower <= upper do
    index = math.floor((lower+upper)/2)

    if self.link_list[index].innovation == gene.innovation then
      -- neuron is already on the list
      return
    elseif self.link_list[index].innovation > gene.innovation then
      upper = index-1
    elseif self.link_list[index].innovation < gene.innovation then
      lower = index+1
    end
  end

  if self.link_list[index].innovation < gene.innovation then
    table.insert(self.link_list, index+1, gene)
  else
    table.insert(self.link_list, index, gene)
  end
end

function Genome:buildNeuralNetwork()
  local function newLink(input, output, weight)
    local o = {}
    o.input = input
    o.output = output
    o.weight = weight

    return o
  end

  local function newNeuron(id, neuron_type, response, y)
    local o = {}
    o.id = id -- the neuron gene id
    o.incoming = {}
    o.leaving = {}
    o.sum = 0
    o.output = 0
    o.neuron_type = neuron_type
    o.response = response
    o.y = y

    return o
  end

  local neuron_list = {}

  -- first create the neurons:
  for n=1,#self.neuron_list do
    local neuron = self.neuron_list[n]

    table.insert(neuron_list, newNeuron(neuron.id, neuron.neuron_type, neuron.response, neuron.y))
    --print("id=".. neuron.id, "type=".. neuron.neuron_type, "response=".. neuron.response, "y=".. neuron.y)
    --print()
  end

  -- now create the links
  for n=1,#self.link_list do
    local gene = self.link_list[n]

    if gene.enabled then
      local link = newLink(gene.from, gene.to, gene.weight)
      --print("link from ".. gene.from .. " to ".. gene.to)

      local _,i1 = self:getNeuron(gene.from)
      local _,i2 = self:getNeuron(gene.to)

      --print("indexes ".. i1 .. " and ".. i2)

      local from, to = neuron_list[i1], neuron_list[i2]

      table.insert(from.leaving, link)
      table.insert(to.incoming, link)
    end
  end

  table.sort(neuron_list,
  function(a, b)
    if a.y == b.y then
      return a.id < b.id
    end
    return a.y > b.y
  end)

  return NeuralNetwork(self.number_of_inputs-self.bias, 1==self.bias, self.number_of_outputs, neuron_list, "sigmoid", self)--inputs, bias, outputs, neuron_list, genome
end

function Genome:copy()
  local genome
  local neuron_list = {}
  local link_list = {}

  for n=1,#self.neuron_list do
    table.insert(neuron_list, self.neuron_list[n]:copy())
  end

  for n=1,#self.link_list do
    table.insert(link_list, self.link_list[n]:copy())
  end

  genome = Genome(self.id, neuron_list, link_list)
  genome.species = self.species
  genome.fitness = self.fitness
  genome.number_of_inputs = self.number_of_inputs
  genome.number_of_outputs = self.number_of_outputs
  genome.bias = self.bias

  for k,v in pairs(self.mutation_rates) do
    genome.mutation_rates[k]=v
  end

  return genome
end

function Genome:printNeuronList()
  -- for debug purposes
  for n=1,#self.neuron_list do
    self.neuron_list[n]:print();print()
  end
end

function Genome:printLinkList()
  -- for debug purposes
  for n=1,#self.link_list do
    self.link_list[n]:print();print()
  end
end

function Genome:draw(x, y, width, height)
  local radius = 10

  local function transform(px, py, w, h)
    return px*width+x, py*height+y
  end

  local function color_transition(c1, c2, val, min, max)
    local r, g, b, a
    r = c1[1]*(max-val)/(max-min) + c2[1]*(val-min)/(max-min)
    g = c1[2]*(max-val)/(max-min) + c2[2]*(val-min)/(max-min)
    b = c1[3]*(max-val)/(max-min) + c2[3]*(val-min)/(max-min)
    a = c1[4]*(max-val)/(max-min) + c2[4]*(val-min)/(max-min)

    return r,g,b,a
  end

  local negative_neuron_color = {.15,.15,.25,1}--{242/255, 34/255, 34/255, 1}
  local positive_neuron_color = {242/255, 34/255, 34/255, 1}--{84/255, 140/255, 135/255, 1}

  for _, link in ipairs(self.link_list) do
    local neuron1 = self:getNeuron(link.from)
    local neuron2 = self:getNeuron(link.to)

    local x1, y1 = transform(neuron1.x, neuron1.y, width, height)
    local x2, y2 = transform(neuron2.x, neuron2.y, width, height)

    if link.enabled then
      local r,g,b,a = color_transition(negative_neuron_color, positive_neuron_color, link.weight, -1, 1)
      a = math.abs(link.weight)/LuaNEAT.parameters.initialWeightRange
      love.graphics.setColor(r,g,b,a)
      love.graphics.setLineWidth(1)

      if neuron1.id == neuron2.id then -- link is looped
        love.graphics.circle("line", x1+radius/2, y2-radius/2, radius)
      else
        love.graphics.line(x1, y1, x2, y2)
      end

      ----------------- DEBUGGGGGGGGGG ---------------------------
      --love.graphics.setColor(.4, .4, 1, 1)
      --local val = link.weight
      --val = string.format("%.3f", val)
      --love.graphics.print(val, (x1+x2)/2, (y1+y2)/2)
      ------------------------------------------------------------
    end
  end

  for _, neuron in ipairs(self.neuron_list) do
    local xp, yp = transform(neuron.x, neuron.y, width, height)

    --[[if neuron.ntype ~= "bias" then
      love.graphics.setColor(.8, .8, .8, 1)
    else
      love.graphics.setColor(.9, .9, .9, 1)
    end]]

    love.graphics.setColor(color_transition(negative_neuron_color, positive_neuron_color, neuron.value or 0, -1, 1))
    love.graphics.circle("fill", xp, yp, radius)
    --love.graphics.setColor(1, 1, 1, 1)
    --love.graphics.circle("line", xp, yp, radius)

    ----------------- DEBUGGGGGGGGGG ---------------------------
    --love.graphics.setColor(.8, .1, .1, 1)
    --love.graphics.print(neuron.id, xp-3, yp-5)


    --love.graphics.setColor(.1, .8, .1, 1)
    --local val = neuron.value or 0
    --val = string.format("%.3f", val)
    --love.graphics.print(val, xp+radius+3, yp+radius)
    ------------------------------------------------------------

    love.graphics.setColor(1, 1, 1, 1)
  end
end

--------------------------------
--       NEURON GENE          --
--------------------------------

NeuronGene = {}
NeuronGene.mt = {
  __index = NeuronGene
}

-- neuron gene's constructor
setmetatable(NeuronGene, {__call = function(t, id, neuron_type, activation, recurrent, response, x, y)
    local o = {}
    o.id          = id
    o.neuron_type = neuron_type
    o.activation  = activation
    o.recurrent   = recurrent
    o.response    = response or 1
    o.x           = x
    o.y           = y

    -- ntype (neuron type) can be: "input", "output", "hidden" and "bias"
    -- response widens or contracts the activation function (p)

    return setmetatable(o, NeuronGene.mt)
  end
})

function NeuronGene:alterResponse(step)
  self.response = step*(LuaNEAT.random()*2 - 1)
end

function NeuronGene:copy()
  return NeuronGene(self.id, self.neuron_type, self.activation, self.recurrent, self.response, self.x, self.y)
end

function NeuronGene:print()
  print("ID:\t\t".. self.id .. "\nType:\t\t".. self.neuron_type .. "\nactivation:\t".. self.activation.. "\nrecurrent:\t" ..tostring(self.recurrent) .. "\nresponse:\t".. self.response .. "\nx:\t\t".. self.x .."\ny:\t\t".. self.y)
end

--------------------------------
--          LINK GENE         --
--------------------------------

LinkGene = {}
LinkGene.mt = {
  __index = LinkGene
}

-- link gene's constructor
setmetatable(LinkGene, {
  __call = function(t, innovation, weight, from, to, enabled, recurrent)
    local o = {}
    o.innovation  = innovation
    o.weight      = weight
    o.from        = from
    o.to          = to
    o.enabled     = enabled
    o.recurrent   = recurrent

    return setmetatable(o, LinkGene.mt)
  end
})

function LinkGene:randomWeight(range)
  range = range or 1
  self.weight = range*(LuaNEAT.random()*2 - 1)
end

function LinkGene:copy()
  return LinkGene(self.innovation, self.weight, self.from, self.to, self.enabled, self.recurrent)
end

function LinkGene:print()
  print("ID:\t\t".. self.innovation .. "\nfrom:\t\t".. self.from .. "\nto:\t\t".. self.to .. "\nenabled:\t" .. tostring(self.enabled).. "\nrecurrent:\t".. tostring(self.recurrent) .."\nweight:\t\t".. self.weight)
end

--------------------------------
--         INNOVATION         --
--------------------------------

InnovationList = {}
InnovationList.mt = {
  __index = InnovationList,
}

setmetatable(InnovationList, {
  __call = function(t)
    local o = {}
    o.neuron_counter = 0
    o.innovation_counter = 0
    o.neurons = {}
    o.links = {}
    return setmetatable(o, InnovationList.mt)
  end
})

function InnovationList:initialize(inputs, outputs, noBias, hidden_layers)
  if not noBias then
    inputs = inputs + 1
  end

  local hidden = 0

  if #hidden_layers==0 then
    for o=1,outputs do
      for i=1,inputs do
        self:newLink(i, inputs+o)
      end
    end
  else
    local hidden_neurons_id_counter = inputs+outputs+1

    for n=1,#hidden_layers do
      hidden = hidden + hidden_layers[n]
    end

    local function numNeuronsBeforeLayer(j)
      if j==1 then return inputs end

      local count = inputs

      for n=1, j-1 do
        count = count+hidden_layers[n]
      end

      return count
    end

    for l=1, #hidden_layers do
      for n=1, hidden_layers[l] do
        if l==1 then
          for j=1, inputs do
            self:newLink(j, hidden_neurons_id_counter)--innovation, weight, from, to, enabled, recurrent
          end
        else
          for j=1, hidden_layers[l-1] do
            local num_so_far = numNeuronsBeforeLayer(l-1)
            self:newLink(num_so_far+outputs+j, hidden_neurons_id_counter)--innovation, weight, from, to, enabled, recurrent
          end
        end

        hidden_neurons_id_counter = hidden_neurons_id_counter + 1
      end
    end

    local num_so_far = numNeuronsBeforeLayer(#hidden_layers)
    for n=1,outputs do
      for k=1,hidden_layers[#hidden_layers] do
        self:newLink(num_so_far+outputs+k, inputs+n)--innovation, weight, from, to, enabled, recurrent
      end
    end
  end

  self.neuron_counter = inputs+outputs+hidden
end

function InnovationList:newNeuron(from, to)
  self.neuron_counter = self.neuron_counter + 1

  if self.neurons[from] == nil then
    self.neurons[from] = {}
  end

  table.insert(self.neurons[from],{
    id = self.neuron_counter,
    to = to
  })

  return self.neuron_counter
end

function InnovationList:getNeuronID(from, to)
  local list = self.neurons[from]
  if list then
    for n=1, #list do
      if list[n].to == to then
        return list[n].id
      end
    end
  end
end

function InnovationList:newLink(from, to)
  -- inserts a new link innovation on innovation list
  self.innovation_counter = self.innovation_counter + 1

  if self.links[from] == nil then
    self.links[from] = {}
  end

  table.insert(self.links[from],{
    innovation = self.innovation_counter,
    to = to
  })

  return self.innovation_counter
end

function InnovationList:getLinkInnovation(from, to)
  local list = self.links[from]
  if list then
    for n=1, #list do
      if list[n].to == to then
        return list[n].innovation
      end
    end
  end
end

function InnovationList:printNeurons()
  for from, links in pairs(self.neurons) do
    for n=1, #links do
      print("id ".. links[n].id .. "\nfrom: ".. from .. "\nto: ".. links[n].to .. "\n")
    end
  end
end

function InnovationList:printLinks()
  for from, links in pairs(self.links) do
    for n=1, #links do
      print("innovation ".. links[n].innovation .. "\nfrom: ".. from .. "\nto: ".. links[n].to .. "\n")
    end
  end
end

function InnovationList:reset(inputs, outputs, noBias, hidden_layers)
  local neuron_counter = self.neuron_counter
  local innovation_counter = self.innovation_counter
  self.neuron_counter = 0
  self.innovation_counter = 0
  self.neurons = {}
  self.links = {}

  self:initialize(inputs, outputs, noBias, hidden_layers)
  self.neuron_counter = self.neuron_counter
  self.innovation_counter = self.innovation_counter
end

--------------------------------
--           SPECIES          --
--------------------------------

Species = {}
Species.mt = {__index = Species}

setmetatable(Species, {
__call = function(t, id, leader)
  local o = {}

  o.id = id
  o.genomes = {}
  o.leader = leader
  o.best_fitness_yet = 0
  o.average_fitness = 0
  o.staleness = 0 -- number of generations without improvement
  o.spawn_amount = 0

  return setmetatable(o, Species.mt)
end})

function Species:breed(id, parameters, innovation_list, layers)
  local genome1, genome2, offspring, index

  genome1, index = self:tournamentSelection(parameters.tournamentSize)

  if LuaNEAT.random() < parameters.crossoverRate and #self.genomes > 1 then
    -- swaping genome1 with this species' first genome
    -- it doesn't matter swap then again after crossover
    -- because genomes will be ordered in the culling process.
    self.genomes[1], self.genomes[index] = self.genomes[index], self.genomes[1]

    genome2 = self:tournamentSelection(parameters.tournamentSize, 1)
    offspring = Genome.crossover(genome1, genome2, layers, parameters)
  else
    -- copying genome 1
    offspring = genome1:copy()
  end

  offspring.id = id
  offspring.fitness = 0

  offspring:mutate(parameters, innovation_list)

  return offspring
end

function Species:tournamentSelection(size, shift)
  local index
  local best

  for n=1, size do
    local i = LuaNEAT.random(1+(shift or 0), #self.genomes)

    if not best then
      best = self.genomes[i]
      index = i
    else
      if self.genomes[i].fitness > best.fitness then
        best = self.genomes[i]
        index = i
      end
    end
  end

  return best, index
end

function Species:getBestGenome()
  local best = self.genomes[1]

  for n=2, #self.genomes do
    if self.genomes[n].fitness > best.fitness then
      best = self.genomes[n]
    end
  end

  return best
end

--------------------------------
--       NEURAL NETWORK       --
--------------------------------

NeuralNetwork = {}
NeuralNetwork.mt = {
  __index = NeuralNetwork,
}

setmetatable(NeuralNetwork, {
  __call = function(t, inputs, bias, outputs, neuron_list, activation, genome)
    local o = {}

    o.inputs = inputs
    o.bias = bias -- a boolean value
    o.outputs = outputs
    o.depth = 1
    o.neuron_list = neuron_list or {}
    o.activation = activation
    o.genome = genome -- a reference to the genome that created this net

    return setmetatable(o, NeuralNetwork.mt)
  end
})

function NeuralNetwork:forward(...)
  local inputs={...}
  local run_type

  -- handling inputs
  if type(inputs[1]) == "string" then
    run_type = inputs[1]
    table.remove(inputs, 1)

    if run_type ~= "active" and run_type ~= "snapshot" then
      error("run_type must either be active or snapshot", 2)
    end
  end

  if type(inputs[1]) == "table" then
    inputs = inputs[1]
  end

  if #inputs ~= self.inputs then
    error("wrong number of inputs ("..#inputs .."); expected ".. self.inputs, 2)
  end

  -- if there is a bias neuron we insert 1 into the input array
  if self.bias then
    table.insert(inputs, 1)
  end

  -- feed forwarding the network

  local flush_count = 1

  if run_type == "snapshot" then
    flush_count = self.depth
  end

  local neurons = self.neuron_list

  -- inserting inputs
  for n=1, #inputs do
    neurons[n].output = inputs[n]

    ----------------------
    ------- DEBUG --------
    local neur = self.genome:getNeuron(neurons[n].id)
    neur.value = inputs[n]
    ----------------------
    ----------------------
  end

  local outputs

  for n=1, flush_count do
    outputs = {}

    for i = #inputs+1, #neurons do
      ---------------------------print("EVALUATING NEURON ".. neurons[i].id .. "; type is '".. neurons[i].neuron_type .."'")
      local sum=0

      -- now we sum all the values incoming to this neuron
      for j=1, #neurons[i].incoming do
        local link = neurons[i].incoming[j]
        local value = self:getNeuron(link.input).output
        local weight = link.weight
        ----------------------------------------print(string.format("summing from neuron %d; value = %.2f; connection weight = %.2f", self:getNeuron(link.input).id, value, weight))

        sum = sum + value*weight
      end
      ------------------------------print(string.format("total sum is %.3f", sum))

      if #neurons[i].incoming > 0 then
        neurons[i].output = activations["sigmoid"](sum, 4.9)--activations[self.activation](sum, neurons[i].response)
        ------------------------------------------------------------------------print("value to register on neuron ".. neurons[i].id .. " is ".. neurons[i].output)
        ----------------------
        ------- DEBUG --------
        local neur = self.genome:getNeuron(neurons[i].id)
        neur.value = neurons[i].output
        ----------------------
        ----------------------
      else
        ------------------------------------print("value to register on neuron ".. neurons[i].id .. " is ".. neurons[i].output .. " (this neuron has no incoming connections)")
      end

      if neurons[i].neuron_type == "output" then
        table.insert(outputs, neurons[i].output)
        ---------------------------------print("inserting output #".. #outputs)
      end
      ------------------------------------------print("next\n\n")
    end
  end

  -- clear all neurons in case this is a snapshot update
  -- so we dont create dependencies
  if run_type == "snapshot" then
    for i=#inputs+1, #neurons do
      neurons[i].output = 0
    end
  end

  return outputs
end

function NeuralNetwork:getNeuron(id)
  for n=1,#self.neuron_list do
    if self.neuron_list[n].id == id then
      return self.neuron_list[n]
    end
  end
end

-- NeuralNetwork's public methods

function NeuralNetwork:draw(x, y, width, height)
  self.genome:draw(x, y, width, height)
end

function NeuralNetwork:getFitness()
  return self.genome.fitness
end

function NeuralNetwork:setFitness(fitness)
  if fitness < 0 then error("negative fitness", 2) end
  self.genome.fitness = fitness
end

function NeuralNetwork:incrementFitness(inc)--etFitness(fitness)
  --if fitness < 0 then error("negative fitness", 2) end
  --self.genome.fitness = fitness
  self.genome.fitness = self.genome.fitness+inc
end

function NeuralNetwork:getGenomeID()
  return self.genome.id
end

function NeuralNetwork:getSpeciesID()
  return self.genome.species.id
end

function NeuralNetwork:printNeuronList()
  for _, neuron in ipairs(self.neuron_list) do
    print("id:\t".. neuron.id)
    print("incoming to this neuron:\t")
    for _, link in ipairs(neuron.incoming) do
      print(link.input .. " >> ".. link.output)
    end
    print("leaving from this neuron:\t")
    for _, link in ipairs(neuron.leaving) do
      print(link.input .. " >> ".. link.output)
    end
    print()
  end
  --[[local function newLink(input, output, weight)
    local o = {}
    o.input = input
    o.output = output
    o.weight = weight

    return o
  end

  local function newNeuron(id, neuron_type, response, y)
    local o = {}
    o.id = id -- the neuron gene id
    o.incoming = {}
    o.leaving = {}
    o.sum = 0
    o.output = 0
    o.neuron_type = neuron_type
    o.response = response
    o.y = y

    return o
  end]]
end

function NeuralNetwork:printLinkList()
end

------------------------
--         POOL       --
------------------------

Pool = {}
Pool.mt = {
  __index = Pool,
}

setmetatable(Pool, {
  __call = function(t, size, inputs, outputs, noBias)
    local o = {}

    o.size = size
    o.inputs = inputs
    o.outputs = outputs
    o.hidden_layers = {}
    o.noBias = noBias or false
    o.species = {}
    o.nets = {}
    o.last_best = nil -- the best performing genome from the previous generation
    o.top_fitness = 0
    o.generation = -1
    o.genome_counter = 0
    o.species_counter = 0
    o.collect_stats = true
    o.statistics = Statistics()
    o.innovation_list = InnovationList()
    o.keep_innovations_forever = false
    o.parameters = {}

    -- cloning default parameters
    for k,v in pairs(LuaNEAT.parameters) do
      if k=="mutation_rates" then
        o.parameters[k] = {}
        for k2,v2 in pairs(LuaNEAT.parameters.mutation_rates) do
          o.parameters[k][k2] = v2
        end
      else
        o.parameters[k] = v
      end
    end

    return setmetatable(o, Pool.mt)
  end
})

function Pool:speciate(genome)
  -- inserts the genome into a compatible species
  for n=1, #self.species do
    local species = self.species[n]
    local leader = species.leader

    if Genome.sameSpecies(genome, leader, self.parameters) then
      -- a new species was found
      genome.species = self.species[n]--species.id
      table.insert(species.genomes, genome)
      return 1
    end
  end

  -- no species is compatible with genome or there are no species yet
  -- so this is a new species
  self.species_counter = self.species_counter + 1
  local id = self.species_counter
  local species = Species(id, genome)
  genome.species = species

  table.insert(species.genomes, genome)
  table.insert(self.species, species)

  return 0
end

function Pool:removeStaleSpecies()
  local survived = {}

  for n=1, #self.species do
    local species = self.species[n]
    local best = species:getBestGenome()

    if best.fitness > species.best_fitness_yet then
      species.best_fitness_yet = best.fitness
      species.staleness = 0
    else
      species.staleness = species.staleness + 1
    end

    if species.staleness < self.parameters.maxStaleness or best.fitness >= self.top_fitness then
      table.insert(survived, species)
    end
  end

  if #survived == 0 then return; end

  self.species = survived
end

function Pool:removeWeakSpecies()
  local survived = {}

  for n=1, #self.species do
    local species = self.species[n]
    if species.spawn_amount >= 1 then
      table.insert(survived, species)
    end
  end

  if #survived == 0 then return; end

  self.species = survived
end

function Pool:getAverageAdjustedFitness()
  local total_sum = 0
  for n=1, #self.species do
    local adj_sum = 0
    for k=1, #self.species[n].genomes do
      adj_sum = adj_sum + self.species[n].genomes[k].fitness
    end
    adj_sum = adj_sum/#self.species[n].genomes
    total_sum = total_sum + adj_sum
  end

  return total_sum/self.size
end

function Pool:calculateSpawnLevels()
  local average = self:getAverageAdjustedFitness()

  for n=1,#self.species do
    local spawn_amount = 0
    for k=1, #self.species[n].genomes do
      local adjusted_fitness = self.species[n].genomes[k].fitness/#self.species[n].genomes
      spawn_amount = spawn_amount + adjusted_fitness
    end
    self.species[n].spawn_amount = spawn_amount/average
  end
end

function Pool:cullSpecies(cutToOne)
  for n=1,#self.species do
    local species = self.species[n]
    -- sorting genomes in the species
    table.sort(species.genomes,
    function(a, b)
      return a.fitness > b.fitness
    end)
    -- done sorting

    local remaining = math.ceil(#species.genomes/2)
    if cutToOne then remaining = 1; end

    while #species.genomes > remaining do
      table.remove(species.genomes)
    end
  end
end

function Pool:getRandomSpecies()
  local index = LuaNEAT.random(1, #self.species)

  return self.species[index]
end

-- Pool's public methods --

function Pool:setInitialHiddenLayers(...)
  local layers = {...}

  if type(layers[1]) == "table" then
    layers = layers[1]
  end

  self.hidden_layers = layers
end

-- initializes the pool
function Pool:initialize(disconnected)
  if self.generation ~= -1 then error("pool has already been initialized", 2) end
  self.innovation_list:initialize(self.inputs, self.outputs, self.noBias, self.hidden_layers)

  for n=1, self.size do
    self.genome_counter = self.genome_counter + 1

    local id = self.genome_counter
    local genome

    if #self.hidden_layers == 0 then
      genome = Genome.minimal(id, self.inputs, self.outputs, self.parameters, copy_mutation_rates(self.parameters), self.noBias, disconnected)
    else
      genome = Genome.layered(id, self.inputs, self.outputs, self.hidden_layers, self.parameters, copy_mutation_rates(self.parameters), self.noBias, disconnected)
    end

    if disconnected then
      for n=1, math.max(1, self.parameters.mutation_rates.addLink) do
        genome:newLink(self.parameters, genome.mutation_rates, self.innovation_list, true)
      end
    end

    table.insert(self.nets, genome:buildNeuralNetwork())
    self:speciate(genome)
  end

  self.generation = 1
end

-- creates the next generation
function Pool:nextGeneration()
  -- deleting previously created neural nets
  self.nets = {}
  -- stats
  local fitness_average = 0
  local fitnesses = {} -- inserting all fitnesses to a table so we can sort it and get the median fitness

  -- getting fitness stats
  if self.collect_stats then
    for s=1,#self.species do
      for g=1,#self.species[s].genomes do
        fitness_average = fitness_average + self.species[s].genomes[g].fitness
        table.insert(fitnesses, self.species[s].genomes[g].fitness)
      end
    end
    fitness_average = fitness_average/self.size
  end

  -- calculating variance
  local fitness_variance = 0

  if self.collect_stats then
    for s=1,#self.species do
      for g=1,#self.species[s].genomes do
        fitness_variance = fitness_variance + (self.species[s].genomes[g].fitness - fitness_average)^2
      end
    end
    fitness_variance = fitness_variance/self.size
  end


  -- removing stale and weak species and calculating the spawn levels of the surviving ones
  local species_amount = #self.species
  local weak_removed, stale_removed

  self:cullSpecies()
  self:removeStaleSpecies(); stale_removed = species_amount - #self.species;    species_amount=#self.species
  self:calculateSpawnLevels()
  self:removeWeakSpecies(); weak_removed = species_amount - #self.species

  -- spawning offspring
  local spawn = {}

  for n=1, #self.species do
    local species = self.species[n]

    for n = 1, math.floor(species.spawn_amount)-1 do
      self.genome_counter = self.genome_counter + 1
      local offspring = species:breed(self.genome_counter, self.parameters, self.innovation_list, self.layers)
      table.insert(spawn, offspring)
    end
  end

  while (#self.species + #spawn) < self.size do
    local species = self:getRandomSpecies()
    self.genome_counter = self.genome_counter + 1
    local offspring = species:breed(self.genome_counter, self.parameters, self.innovation_list)

    table.insert(spawn, offspring)
  end

  -- deleting all genomes except for the best performing one in each species
  self:cullSpecies(true)

  -- setting the new species' leader as its best genome (genomes are always ranked after culling)
  -- and getting the best genome from last generation
  local best = self.species[1].genomes[1]
  for n=1,#self.species do
    local leader = self.species[n].genomes[1]
    self.species[n].leader = leader

    if leader.fitness > best.fitness then
      best = leader
    end
  end
  self.last_best = best
  self.top_fitness = best.fitness

  -- saving new stats
  if self.collect_stats then
    local fitness_median

    table.sort(fitnesses)
    if math.fmod(#fitnesses,2) == 0 then
      fitness_median = (fitnesses[#fitnesses/2] + fitnesses[(#fitnesses/2)+1]) / 2
    else
      fitness_median = fitnesses[math.ceil(#fitnesses/2)]
    end

    self.statistics:new(self.top_fitness, fitness_average, fitness_median, fitness_variance)
  end

  -- speciating the offsprings
  local species_created = 0
  for n=1, #spawn do
    species_created = species_created + self:speciate(spawn[n])
  end

  -- building the neural nets
  for n=1,#self.species do
    -- reseting all leaders' fitness
    self.species[n].genomes[1].fitness=0--leader.fitness = 0

    for k=1,#self.species[n].genomes do
      local genome = self.species[n].genomes[k]
      table.insert(self.nets, genome:buildNeuralNetwork())
    end
  end

  self.generation = self.generation + 1

  if not self.keep_innovations_forever then
    self.innovation_list:reset(self.inputs, self.outputs, self.noBias, self.hidden_layers)
  end

  return "Generation ".. self.generation-1 .. "\n".. weak_removed .. " weak species removed;\n ".. stale_removed .. " stale species removed;\n".. species_created .."species created;\nBest fitness was ".. self.top_fitness
end

-- returns the current generation
function Pool:getGeneration()
  return self.generation
end

-- returns how many species are in the pool
function Pool:getSpeciesAmount()
  return #self.species
end

-- returns all the neural nets so we can train them
function Pool:getNeuralNetworks()
  if #self.nets == 0 then error("Pool must be initialized first", 2) end
  return self.nets
end

-- returns the current best neural net
function Pool:getBestPerformer()
  local best = self.species[1].genomes[1]

  for s=1,#self.species do
    for g=1,#self.species[s].genomes do
      local current = self.species[s].genomes[g]

      if current.fitness > best.fitness then
        best = current
      elseif current.fitness == best.fitness then
        if #current.link_list < #best.link_list then
          best = current
        end
      end
    end
  end

  return best:buildNeuralNetwork()
end

-- returns the best neural net from the previous generation
function Pool:getLastBestPerformer()
  return self.last_best:buildNeuralNetwork()
end

-- returns the best fitness from previous generation
function Pool:getLastBestFitness()
  return self.top_fitness
end

-- enables or disables statistics collection
function Pool:collectStatistics(collect)
  self.collect_stats = (collect == true)
end

function Pool:getTopFitnessPoints()
  return self.statistics:getTopFitnessPoints()
end

--------------------------------
--        STATISTICS          --
--------------------------------

Statistics = {}
Statistics.mt = {__index=Statistics}

setmetatable(Statistics, {
  __call = function(t)
    return setmetatable({}, Statistics.mt)
  end
})

function Statistics:new(top_fitness, fitness_average, fitness_median, fitness_variance)
  table.insert(self,{
    ["top_fitness"] = top_fitness,
    ["fitness_average"] = fitness_average,
    ["fitness_median"] = fitness_median,
    ["fitness_variance"] = fitness_variance,
  })
end

function Statistics:getTopFitnessPoints()
  local points={}

  for n=1,#self do
    table.insert(points, self[n].top_fitness)
  end

  return points
end

function Statistics:getAllTimeTopFitness()
  local max = 0

  for n=1,#self do
    if self[n].top_fitness > max then
      max = self[n].top_fitness
    end
  end

  return max
end

function Statistics:getAverageFitnessPoints()
  local points={}

  for n=1,#self do
    table.insert(points, self[n].fitness_average)
  end

  return points
end

function Statistics:getMedianFitnessPoints()
  local points={}

  for n=1,#self do
    table.insert(points, self[n].fitness_median)
  end

  return points
end

function Statistics:getFitnessVariancePoints()
  local points={}

  for n=1,#self do
    table.insert(points, self[n].fitness_variance)
  end

  return points
end

--------------------------------
--      PUBLIC FUNCTIONS      --
--------------------------------

function LuaNEAT.newPool(size, inputs, outputs, bias)
  -- error handling: size, inputs and outputs needs to be integers greater than 0

  -- handling size
  if type(size)~="number" then
    error("number expected, got ".. type(size), 2)
  end
  if size<=0 then
    error("size needs to be greater than 0", 2)
  end
  if math.floor(size)~=size then
    error("size needs to be a an integer", 2)
  end

  -- handling inputs
  if type(inputs)~="number" then
    error("number expected, got ".. type(inputs), 2)
  end
  if inputs<=0 then
    error("inputs needs to be greater than 0", 2)
  end
  if math.floor(inputs)~=inputs then
    error("inputs needs to be an integer", 2)
  end

  -- handling outputs
  if type(outputs)~="number" then
    error("number expected, got ".. type(outputs), 2)
  end
  if outputs<=0 then
    error("outputs needs to be an integer", 2)
  end
  if math.floor(outputs)~=outputs then
    error("outputs needs to be an integer", 2)
  end

  local noBias = (bias == false)

  return Pool(size, inputs, outputs, noBias)
end

function LuaNEAT.save(pool, filename, love2d)
  local open = io.read
  local write = io.write
  if love2d then
  end

  local file,error = io.open(filename..".txt", "w+")
  if not file then
    print("cannot open file: ".. error)
    return
  end

  -- saving order:
  --  pool info
  --  pool parameters
  --  innovation list info
  --  neuron innovations
  --  link innovations
  --  species


  -- pool info
  file:write("POOL\n")
  for k,v in pairs(pool) do
    if type(v) ~= "table" then
      file:write(k .. ":".. tostring(v).. "\n")
    end
  end

  -- pool parameters
  file:write("PARAMETERS\n")
  for k,v in pairs(pool.parameters) do
    file:write(k .. ":".. tostring(v).. "\n")
  end

  -- innovation list info
  file:write(
    "INNOVATION LIST\n" ..

    "neuron_counter:" .. pool.innovation_list.neuron_counter .. "\n" ..
    "innovation_counter:" .. pool.innovation_list.neuron_counter .. "\n"
  )

  -- neuron innovations
  for from,list in ipairs(pool.innovation_list.neurons) do
    for n=1,#list do
      file:write(
        "NEURON INNOVATION".."\n" ..
        "id:".. list[n].id .."\n" ..
        "from:".. from .."\n"..
        "to:".. list[n].to .."\n"
      )
    end
  end

  -- link innovations
  for from,list in ipairs(pool.innovation_list.links) do
    for n=1,#list do
      file:write(
        "LINK INNOVATION".."\n" ..
        "innovation:".. list[n].innovation .."\n" ..
        "from:".. from .."\n"..
        "to:".. list[n].to .."\n"
      )
    end
  end

  -- species
  for s=1,#pool.species do
    file:write("SPECIES\n")
    for k, v in pairs(pool.species[s]) do
      if type(v) ~= "table" then
        file:write(k..":"..v.."\n")
      end
    end
  end



  --o.neuron_counter = 0
  --o.innovation_counter = 0

  file:close()
end

function LuaNEAT.load(filename)
end

function LuaNEAT.version()
  return LuaNEAT._VERSION
end

return LuaNEAT
