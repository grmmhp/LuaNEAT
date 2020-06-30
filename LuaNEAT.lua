local LuaNEAT = {
  _VERSION = "LuaNEAT Alpha",
  _DESCRIPTION = "NEAT module for Lua",
  _URL = "https://github.com/grmmhp/LuaNEAT",
  _LICENSE = [[
    MIT License

    Copyright (c) 2020 Gabriel Mesquita

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

math.randomseed(os.time())
LuaNEAT.random = love.math.random--math.random

--TODO:
-- rewrite NeuralNetwork:getNeuron() (copy Genome:getNeuron() code)
-- write activation functions
-- calculate network depth
-- pass activation function to neural net

--at the end of development:
-- remove activation from NeuronGene
-- remove math.randomseed()
-- check if always using LuaNEAT.random()

--------------------------------
--         PARAMETERS         --
--------------------------------

LuaNEAT.parameters = {
  -- general parameters
  defaultActivation = "sigmoid",
  findLinkAttempts = 20,
  hiddenNeuronsThreshold = 5,
  maxNeuronsAmount = 1e6,

  -- speciation
  excessGenesCoefficient   = 1,
  disjointGenesCoefficient = 1,
  matchingGenesCoefficient = .4,

  sameSpeciesThreshold     = 1,
  generationsNoImprovement = 15,
  tournamentSize           = 3,

  youngAgeBonusThreshold = 10,
  youngAgeFitnessBonus   = 1.3,
  oldAgeThreshold        = 50,
  oldAgePenalty          = .7,

  -- mutation parameters
  weightLimit = 2,
  responseLimit = 2,

  -- mutation rates
  addLink           = .07,
  addNode           = .03,
  loopedLink        = .05,
  enableDisable     = .01,

  perturbWeight   = .5,
  replaceWeight   = .1,
  maxPerturbation = .1,
  perturbResponse = .1,
  alterResponse   = .2,
  maxResponse     = 2,

  -- crossover
  crossoverRate = .75,
}


--------------------------------
--    ACTIVATION FUNCTIONS    --
--------------------------------

local activations = {
  ["sigmoid"] = function(x,p)
    return 1/(1+math.exp(-x*p))
  end,

  ["sine"] = function(x,p)
    return 1
  end,

  ["ReLU"] = function(x,p)
    return 1--math.max(0, math.min(x, 1))
  end,
}


-- forward declaration of classes
local Genome, NeuronGene, LinkGene, NeuralNetwork, InnovationList, Species, Pool

--------------------------------
--           GENOME           --
--------------------------------

Genome = {}
Genome.mt = {
  __index = Genome,
}

setmetatable(Genome, {
  __call = function(t, id, neuron_list, link_list)
    local o = {}

    o.id = id or -1
    o.species = -1
    o.fitness = 0
    o.adjustedFitness = 0
    o.number_of_inputs = 0
    o.number_of_outputs = 0
    o.bias = 0
    o.neuron_list = neuron_list or {}
    o.link_list = link_list or {}

    return setmetatable(o, Genome.mt)
  end
})

function Genome.minimal(id, inputs, outputs, parameters, noBias)
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
    local gene = NeuronGene(n, "input", parameters.defaultActivation, false, 1, n*dx, 1)
    table.insert(neuron_list, gene) --id, neuron_type, activation, recurrent, response, x, y
  end

  if not noBias then
    local gene = NeuronGene(inputs+1, "bias", parameters.defaultActivation, false, 1, (inputs+1)*dx, 1)
    table.insert(neuron_list, gene)
    inputs = inputs + 1
  end

  dx = 1/(outputs+1)
  for n=1,outputs do
    local gene = NeuronGene(inputs+n, "output", parameters.defaultActivation, false, 1, n*dx, 0)
    table.insert(neuron_list, gene)

    -- creating links
    for k=1,inputs do
      local innovation = k + (inputs)*(n-1)
      local link = LinkGene(innovation, 0, k, inputs+n, true, false)--innovation, weight, from, to, enabled, recurrent
      link:randomWeight(parameters.weightLimit)

      table.insert(link_list, link)
    end
  end

  genome = Genome(id, neuron_list, link_list)
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

  print("last innovations:")
  print(genome1.link_list[#genome1.link_list].innovation)
  print(genome2.link_list[#genome2.link_list].innovation)

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
        offspring:insertLink(link2:copy(), true)
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

  return offspring
end

function Genome:newNode(parameters, innovation_list)
  if #self.neuron_list >= parameters.maxNeuronsAmount then return end

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
    while true do
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
      1,
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
      1,
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

function Genome:newLink(parameters, innovation_list, noLoop)
  local from, to
  local recurrent = false

  if LuaNEAT.random() < parameters.loopedLink and not noLoop then
    -- a looped link will be created
    -- selects a hidden neuron to be selected for a looped link

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
  if from.y < to.y then recurrent=true end

  -- innovation, weight, from, to, enabled, recurrent
  local gene = LinkGene(id, 0, from.id, to.id, true, recurrent)
  gene:randomWeight(parameters.weightLimit)

  self:insertLink(gene)
  return "successfully created new link between nodes ".. from.id .. " and ".. to.id
end

function Genome:mutate()
  if LuaNEAT.random() < parameters.addNode then
    self:newNode()
  end

  if LuaNEAT.random() < parameters.addLink then
    self:newLink()
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
  local lower, upper = 1, #self.neuron_list
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

  local function newNeuron(id, neuron_type, response)
    local o = {}
    o.id = id -- the neuron gene id
    o.incoming = {}
    o.leaving = {}
    o.sum = 0
    o.output = 0
    o.neuron_type = neuron_type
    o.response = response

    return o
  end

  local neuron_list = {}

  -- first create the neurons:
  for n=1,#self.neuron_list do
    local neuron = self.neuron_list[n]
    table.insert(neuron_list, newNeuron(neuron.id, neuron.neuron_type, neuron.response))
  end

  -- now create the links
  for n=1,#self.link_list do
    local gene = self.link_list[n]

    if gene.enabled then
      local link = newLink(gene.from, gene.to, gene.weight)

      local _,i1 = self:getNeuron(gene.from)
      local _,i2 = self:getNeuron(gene.to)

      local from = neuron_list[i1]
      local to = neuron_list[i2]

      table.insert(from.leaving, link)
      table.insert(to.incoming, link)
    end
  end

  return NeuralNetwork(self.number_of_inputs-self.bias, 1==self.bias, self.number_of_outputs, neuron_list, "sigmoid", self)--inputs, bias, outputs, neuron_list, genome
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

function LinkGene:randomWeight(limit)
  limit = limit or 1
  self.weight = LuaNEAT.random()*2*limit - limit
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

function InnovationList:initialize(inputs, outputs, noBias)
  if not noBias then
    inputs = inputs + 1
  end

  for o=1,outputs do
    for i=1,inputs do
      self:newLink(i, inputs+o)
    end
  end

  self.neuron_counter = inputs+outputs
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
  o.best_fitnessSoFar = 0
  o.average_fitness = 0
  o.staleness = 0
  o.age = 0
  o.spawn_amount = 0

  return setmetatable(o, Species.mt)
end})

function Species:adjustFitnesses(parameters)
  -- "this method boosts the fitnesses of the young, penalizes the
  --  fitnesses of the old and then performs fitness sharing over
  --  all the members of the species"
  for n=1,#self.genomes do
    local genome = self.genomes[n]
    local fitness = 0

    if self.age < parameters.youngAgeBonusThreshold then
      fitness = (genome.fitness)*(parameters.youngAgeFitnessBonus)
    end

    if self.age > parameters.oldAgeThreshold then
      fitness = (genome.fitness)*(parameters.oldAgePenalty)
    end

    genome.adjustedFitness = (fitness)/(#self.genomes)
  end
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

  -- if there is a bias neuron we add 1 to the input table
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
  end

  local outputs

  for n=1, flush_count do
    outputs = {}

    -- first we go from the first hidden neuron (#inputs+#ouputs+1)
    -- to the last hidden one (#neurons)

    for i = #inputs+self.outputs+1, #neurons do
      local sum=0

      -- now we sum all the values incoming to this neuron
      for j=1, #neurons[i].incoming do
        local link = neurons[i].incoming[j]
        local value = self:getNeuron(link.input).output
        local weight = link.weight

        sum = sum + value*weight
      end

      neurons[i].output = activations[self.activation](sum, neurons[i].response)
    end

    -- then from the first output (#inputs+1)
    -- to the last output (#inputs+#outputs)

    for i = #inputs+1, #inputs+self.outputs do
      local sum=0

      -- now we sum all the values incoming to this neuron
      for j=1, #neurons[i].incoming do
        local link = neurons[i].incoming[j]
        local value = self:getNeuron(link.input).output
        local weight = link.weight

        sum = sum + value*weight
      end

      neurons[i].output = activations[self.activation](sum, neurons[i].response)
      table.insert(outputs, neurons[i].output)
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
  local _,index = self.genome:getNeuron(id)
  return self.neuron_list[index]
end

function NeuralNetwork:calculateDepth()
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
    o.noBias = noBias or false
    o.species = {}
    o.nets = {}
    o.last_top = nil -- the last best performing genome
    o.top_fitness = 0
    o.generation = -1
    o.genome_counter = 0
    o.species_counter = 0
    o.innovation_list = InnovationList()
    o.parameters = {}

    -- copying default parameters
    for k,v in pairs(LuaNEAT.parameters) do
      o.parameters[k] = v
    end

    o.innovation_list:initialize()

    return setmetatable(o, Pool.mt)
  end
})

function Pool:initialize()
end

function Pool:nextGeneration()
  self:cullSpecies()
  self:removeStaleSpecies()

  self.generation = self.generation + 1
end

function Pool:calculateSpawnLevels()
  local average = self:getAverageAdjustedFitness()

  for _, species in ipairs(self.species) do
    local spawnAmount = 0
    for __, genome in ipairs(species.genomes) do
      spawnAmount = spawnAmount + genome.adjustedFitness
    end
    species.spawnAmount = spawnAmount/average
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


------------------------
----- TESTING AREA -----
------------------------


-- id, neuron_type, activation, recurrent, response, x, y

local inputs,outputs=2,2

local il = InnovationList()
il:initialize(inputs, outputs)

local genome = Genome.minimal(0, inputs, outputs, LuaNEAT.parameters)
print(genome:newNode(LuaNEAT.parameters, il))
print(genome:newLink(LuaNEAT.parameters, il))
local net = genome:buildNeuralNetwork()

print("--------------------Neurons:\n")
genome:printNeuronList()

print("\n\n--------------------Links:\n")
genome:printLinkList()


print("\n\n--------------------Links:\n")
for _, neuron in ipairs(net.neuron_list) do
  print("Neuron ID ".. neuron.id)
  for _, link in ipairs(neuron.incoming) do
    print("-> from ".. link.input .. " to ".. link.output)
  end
  for _, link in ipairs(neuron.leaving) do
    print("-> from ".. link.input .. " to ".. link.output)
  end
end

io.write("\n{")
for _, neuron in ipairs(net.neuron_list) do
  io.write(neuron.id .. ", ")
end
io.write("}\n")

print("Outputs: ")
outputs = net:forward("active", .5, .75)
for k,v in ipairs(outputs) do
  print(v)
end

--[[neuron_list = {
  NeuronGene(1, "hidden", "ReLU", false, 1, 0, 0),--id, neuron_type, activation, recurrent, response, x, y
  NeuronGene(2, "hidden", "ReLU", false, 1, 0, 0),
  NeuronGene(3, "hidden", "ReLU", false, 1, 0, 0),
  NeuronGene(4, "hidden", "ReLU", false, 1, 0, 0),
  NeuronGene(5, "hidden", "ReLU", false, 1, 0, 0),
}

link_list1 = {
  LinkGene(1, 1, 1, 1, true, false),
  LinkGene(2, 1, 1, 1, true, false),
  LinkGene(3, 1, 1, 1, true, false),
  LinkGene(4, 1, 1, 1, true, false),
  LinkGene(5, 1, 1, 1, true, false),
  LinkGene(6, 1, 1, 1, true, false),
  LinkGene(8, 1, 1, 1, true, false),
  LinkGene(12, 1, 1, 1, true, false),
  LinkGene(13, 1, 1, 1, true, false),
  LinkGene(14, 1, 1, 1, true, false)--innovation, weight, from, to, enabled, recurrent
}

link_list2 = {
  LinkGene(1, 1, 1, 1, true, false),
  LinkGene(2, 1, 1, 1, true, false),
  LinkGene(3, 1, 1, 1, true, false),
  LinkGene(4, 1, 1, 1, true, false),
  LinkGene(7, 1, 1, 1, true, false),
  LinkGene(9, 1, 1, 1, true, false),
  LinkGene(10, 1, 1, 1, true, false),
  LinkGene(11, 1, 1, 1, true, false),
  LinkGene(14, 1, 1, 1, true, false),
  LinkGene(15, 1, 1, 1, true, false)--innovation, weight, from, to, enabled, recurrent
}

local genome1 = Genome(1, neuron_list, link_list1)
local genome2 = Genome(2, neuron_list, link_list2)--Genome.minimal(0, inputs, outputs, LuaNEAT.parameters)

genome1.fitness = -10

local offspring = Genome.crossover(genome1, genome2)
for _,neuron in pairs(offspring.neuron_list) do
  neuron:print();print()
end
print("\n\n")
for _,link in pairs(offspring.link_list) do
  link:print();print()
end

print("\n")]]

--[[for from, link in ipairs(il.links) do
  for n=1,#link do
    print("innovation ".. link[n].innovation .. "\nfrom: ".. from .. "\nto: ".. link[n].to)
    print()
  end
end]]

--il:printNeurons(); print()
--il:printLinks()

-----

return LuaNEAT
