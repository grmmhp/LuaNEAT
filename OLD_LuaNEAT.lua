local LuaNEAT = {
  _VERSION = "LuaNEAT Alpha Development",
  _DESCRIPTION = "NEAT module for Lua",
  _URL = "https://github.com/grmmhp/LuaNEAT",
  _LICENSE = [[
    MIT License

    Copyright (c) 2019 Gabriel Mesquita

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

-- TODO:
-- check if there are any unused parameters
-- code neural network stuff

LuaNEAT.DefaultParameters = {
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

  findOtherGenomeAttempts = 5,

  -- mutations
  findLinkAttempts = 20,
  hiddenNeuronsThreshold = 5,
  maxNeuronsAmount = 1000000,
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

  crossoverRate = .75,
}

local random = math.random

if love then
  if love.math then
    random = love.math.random
  end
end

------------
-- MODULE --
------------

-- forward declaration of classes
local Genome, NeuronGene, LinkGene, NeuralNetwork, Neuron, Link, Innovation, InnovationList, Species, Pool

local function sigmoid(x, p)
  local e = 2.71828182846
  return 1/(1+e^(-x*p))
end

--------------------------------
--           GENOME           --
--------------------------------

Genome = {}
Genome.mt = {
  __index = Genome,

  __tostring = function(self)
    return    "ID:\t\t\t" .. self.id
           .. "\nSpecies:\t\t" .. self.species
           .. "\nAmount of neuron genes:\t" .. #self.NeuronGeneList
           .. "\nAmount of link genes:\t" .. #self.LinkGeneList
  end
}

setmetatable(Genome, {
  __call = function(t, id, NeuronGeneList, LinkGeneList)
    local o = {}

    o.id = id
    o.species = -1
    o.fitness = 0
    o.adjustedFitness = 0
    o.NeuronGeneList = NeuronGeneList or {}
    o.LinkGeneList = LinkGeneList or {}

    return setmetatable(o, Genome.mt)
  end
})

function Genome.basicGenome(id, inputs, outputs, noBias)
  -- if noBias is true, basicGenome won't create bias neuron gene
  local neuronList = {}
  local linkList = {}

  -- dx is used to put input, output and bias neurons evenly spaced on the grid
  local dx

  if noBias then
    dx = 1/(inputs+1)
  else
    dx = 1/(inputs+2)
  end

  for n=1, inputs do
    local neuron = NeuronGene(n, "input", false, 1, n*dx, 1)
    table.insert(neuronList, neuron)
  end

  if not noBias then
    local biasNeuron = NeuronGene(inputs+1, "bias", false, 1, (inputs+1)*dx, 1)
    table.insert(neuronList, biasNeuron)
  end

  dx = 1/(outputs+1)

  for n=1, outputs do
    -- puts the output neurons evenly spaced on the grid
    local id

    if noBias then
      id = inputs+n
    else
      id = inputs+n+1
    end

    local neuron = NeuronGene(id, "output", false, 1, n*dx, 0)
    table.insert(neuronList, neuron)
  end

  return Genome(id, neuronList, linkList)
end

function Genome:buildNeuralNetwork()
  local neuronList = {}
  local outputsList = {}
  local inputs = 0
  local outputs = 0
  local bias

  local function binarySearch(id, list)
    list = list or neuronList

    local lower, upper = 1, #list
    local index

    while lower <= upper do
      index = math.floor((lower+upper)/2)

      if list[index].id > id then
        upper = index-1
      elseif list[index].id < id then
        lower = index+1
      else
        return list[index]
      end
    end

    return binarySearch(id, outputsList)
  end

  local bias = 0

  for _, gene in ipairs(self.NeuronGeneList) do
    local neuron = Neuron(gene.id, gene.ntype, gene.response)
    neuron.y = gene.y

    if gene.ntype == "input" then
      inputs = inputs + 1
    elseif gene.ntype == "bias" then
      bias = 1
    elseif gene.ntype == "output" then
      outputs = outputs + 1
    end

    if gene.ntype == "output" then
      table.insert(outputsList, neuron)
    else
      table.insert(neuronList, neuron)
    end
  end

  --[[for index = inputs+bias+1, inputs+bias+outputs do
    local gene = self.NeuronGeneList[index]
    local neuron = Neuron(gene.id, gene.ntype, gene.response)

    table.insert(outputsList, neuron)
  end]]

  for _, gene in ipairs(self.LinkGeneList) do
    if gene.enabled then
      local from = binarySearch(gene.from)
      local to = binarySearch(gene.to)

      local link = Link(gene.from, gene.to, gene.weight, gene.recurrent) --input, output, weight, recurrent

      table.insert(from.leaving, link)
      table.insert(to.incoming, link)
    end
  end

  -- adding outputs to neuronList
  for _, neuron in ipairs(outputsList) do
    table.insert(neuronList, neuron)
  end

  table.sort(neuronList,
  function(a, b)
    if a.y == b.y then
      return a.id < b.id
    end

    return a.y > b.y
  end)

  return NeuralNetwork(inputs, outputs, bias, neuronList, self)
end

function Genome:draw(width, height, sx, sy)
  local radius = 10

  local function transform(px, py, w, h)
    return px*width+sx, py*height+sy
  end

  for _, link in ipairs(self.LinkGeneList) do
    local neuron1 = self:getNeuron(link.from)
    local neuron2 = self:getNeuron(link.to)

    local x1, y1 = transform(neuron1.x, neuron1.y, width, height)
    local x2, y2 = transform(neuron2.x, neuron2.y, width, height)

    if link.enabled then
      love.graphics.setColor(1, 1, 1, 1)
    else
      love.graphics.setColor(.25, .25, .25, 1)
    end

    if neuron1.id == neuron2.id then -- link is looped
      love.graphics.circle("line", x1+radius/2, y2-radius/2, radius)
    else
      love.graphics.line(x1, y1, x2, y2)
    end
  end

  for _, neuron in ipairs(self.NeuronGeneList) do
    local x, y = transform(neuron.x, neuron.y, width, height)

    if neuron.ntype ~= "bias" then
      --love.graphics.setColor(1, 0, 0, 1)
      love.graphics.setColor(.8, .8, .8, 1)
    else
      --love.graphics.setColor(0, 1, 0, 1)
      love.graphics.setColor(.9, .9, .9, 1)
    end

    love.graphics.circle("fill", x, y, radius)--116/255, 249/255, 246/255
    --love.graphics.setColor(0, 0, 0, 1)
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.circle("line", x, y, radius)

    love.graphics.setColor(1, 1, 1, 1)
    --love.graphics.print(neuron.id, x, y)
  end
end

function Genome.crossover(id, genome1, genome2)
  -- genome1 is always the more fit genome
  -- "In composing the offspring, genes are randomly chosen from either parent at matching genes,
  -- whereas all excess or disjoint genes are always included from the more fit parent"
  local offspring
  local neuronGeneList = {}
  local linkGeneList = {}

  local function binaryInsert(neuron)
    if #neuronGeneList == 0 then
      table.insert(neuronGeneList, neuron); return
    end

    local lower, upper = 1, #neuronGeneList
    local index=0

    while lower <= upper do
      index = math.floor((lower+upper)/2)

      if neuronGeneList[index].id == neuron.id then
        -- neuron is already on the list
        return
      elseif neuronGeneList[index].id > neuron.id then
        upper = index-1
      elseif neuronGeneList[index].id < neuron.id then
        lower = index+1
      end
    end

    if neuronGeneList[index].id < neuron.id then
      table.insert(neuronGeneList, index+1, neuron)
    else
      table.insert(neuronGeneList, index, neuron)
    end
  end

  if genome2.fitness > genome1.fitness then
    -- swap the genomes
    genome1, genome2 = genome2, genome1
  elseif genome1.fitness == genome2.fitness then
    -- equal fitness, choose one at random
    if random() < .5 then
      genome1, genome2 = genome2, genome1
    end
  end

  -- gets the inputs, outputs and bias neurons from the genomes
  for _, neuron in ipairs(genome1.NeuronGeneList) do
    if (neuron.ntype == "input") or (neuron.ntype == "output") or (neuron.ntype == "bias") then
      table.insert(neuronGeneList, neuron:copy())
    else
      break
    end
  end

  for i, gene in ipairs(genome1.LinkGeneList) do
    local neuronFrom
    local neuronTo

    if genome2:linkGeneExists(gene) then
      -- this is a matching gene
      -- we'll select one at random

      if random() < .5 then
        table.insert(linkGeneList, (genome2:getLink(gene.innovation)):copy())

        neuronFrom = genome2:getNeuron((genome2:getLink(gene.innovation)).from)
        neuronTo = genome2:getNeuron((genome2:getLink(gene.innovation)).to)
      else
        table.insert(linkGeneList, gene:copy())

        neuronFrom = genome1:getNeuron(gene.from)
        neuronTo = genome1:getNeuron(gene.to)
      end
    else
      -- this is a disjoint or excess gene
      table.insert(linkGeneList, gene:copy())

      neuronFrom = genome1:getNeuron(gene.from)
      neuronTo = genome1:getNeuron(gene.to)
    end

    binaryInsert(neuronFrom:copy())
    binaryInsert(neuronTo:copy())
  end

  offspring = Genome(id, neuronGeneList, linkGeneList)

  return offspring
end

function Genome.calculateCompatibilityDistance(genome1, genome2, parameters)
  local matching    = 0
  local excess      = 0
  local disjoint    = 0
  local weightDiff  = 0

  local index1 = 1 -- genome 1 index
  local index2 = 1 -- genome 2 index

  while index1 <= #genome1.LinkGeneList and index2 <= #genome2.LinkGeneList do
      local id1 = genome1.LinkGeneList[index1].innovation
      local id2 = genome2.LinkGeneList[index2].innovation

      if id1 == id2 then
        matching = matching + 1
        weightDiff = math.abs(genome1.LinkGeneList[index1].weight - genome2.LinkGeneList[index2].weight)

        index1 = index1 + 1
        index2 = index2 + 1
      elseif id1 < id2 then
        disjoint = disjoint + 1
        index1 = index1 + 1
      elseif id2 < id1 then
        disjoint = disjoint + 1
        index2 = index2 + 1
      end
  end

  if index1 <= #genome1.LinkGeneList then
    excess = #genome1.LinkGeneList-index1 + 1
  end

  if index2 <= #genome2.LinkGeneList then
    excess = #genome2.LinkGeneList-index2 + 1
  end

  local maxIndex = math.max(#genome1.LinkGeneList, #genome2.LinkGeneList)

  return   parameters.excessGenesCoefficient*excess/maxIndex
         + parameters.disjointGenesCoefficient*disjoint/maxIndex
         + parameters.matchingGenesCoefficient*weightDiff/matching;
end

function Genome:mutate(parameters, innovationList)
  self:mutatePerturbWeight(parameters)
  self:mutatePerturbResponse(parameters)
  self:mutateEnableDisableLink(parameters)

  if random() < parameters.addNode then
    self:mutateAddNode(parameters, innovationList)
  end

  if random() < parameters.addLink then
    self:mutateAddLink(parameters, innovationList)
  end
end

function Genome:mutateAddLink(parameters, innovationList, forceForwardLink)
  -- adds a forward, recurrent or looped link
  local neuron1
  local neuron2
  local recurrent = false

  if random() < parameters.loopedLink and not forceForwardLink then
    -- tries to find a hidden or output neuron that does not already have a loopback
    -- looped link is going to be added
    for n=1, parameters.findLinkAttempts do
      neuron1 = self:getRandomNeuron()

      if  (neuron1.ntype ~= "input")
      and (neuron1.ntype ~= "bias")
      and (not neuron1.recurrent) then
        neuron2 = neuron1
        neuron1.recurrent = true

        recurrent = true

        break
      end
    end
  else
    -- tries to find two unconnected neurons
    for n=1, parameters.findLinkAttempts do
      neuron1 = self:getRandomNeuron()
      neuron2 = self:getRandomNeuron()

      --if (not self:linkExists(neuron1, neuron2)) and (neuron1.id ~= neuron2.id) and (neuron2.ntype~="input") and (neuron2.ntype~="bias") then
      if (not self:linkExists(neuron1, neuron2)) and (neuron1.id ~= neuron2.id) and (neuron2.ntype~="input") and (neuron2.ntype~="bias") and not (neuron1.ntype == "output" and neuron1.ntype == "output") then
        break
      else
        neuron1 = nil
        neuron2 = nil
      end
    end
  end

  -- could not find neurons
  if neuron1==nil or neuron2==nil then return; end

  -- checking innovation
  local innovationID = innovationList:getID("new_link", neuron1.id, neuron2.id)

  -- checking if link is recurrent
  if neuron1.y < neuron2.y then
    recurrent = true
  else
    recurrent = false
  end

  if innovationID == -1 then
    -- innovation wasnt discovered yet
    local innovation = Innovation("new_link", neuron1.id, neuron2.id, -1, "none")
    -- id, itype, from, to, neuronID, ntype
    innovationID = innovationList:push(innovation)
    -- innovation was already discovered
  end

  --innovation, weight, from, to, enabled, recurrent
  local linkGene = LinkGene(innovationID, 0, neuron1.id, neuron2.id, true, recurrent)
  linkGene:randomWeight(parameters.weightLimit)

  self:pushLinkGene(linkGene)
end

function Genome:mutateAddNode(parameters, innovationList)
  local link
  local sizeThreshold = self:getBaseNeuronsAmount()+parameters.hiddenNeuronsThreshold

  if #self.LinkGeneList == 0 then return; end
  if #self.NeuronGeneList > parameters.maxNeuronsAmount then return; end

  if #self.LinkGeneList < sizeThreshold then
    -- genome is too small; will choose an old link
    for n=1, parameters.findLinkAttempts do
      link = self:getRandomOldLink()

      local fromNeuron = self:getNeuron(link.from)

      if (link.enabled) and (not link.recurrent) and (fromNeuron.ntype~="bias") then
        break
      else
        link = nil
      end
    end
  else
    -- genome is of sufficient size
    while true do
      link = self:getRandomLink()

      local fromNeuron = self:getNeuron(link.from)

      if (link.enabled) and (not link.recurrent) and (fromNeuron.ntype~="bias") then
        break
      end
    end
  end

  if link == nil then return; end

  -- we will disable this link and create two others
  link.enabled = false

  local neuron    -- the neuron to be added
  local linkFrom  -- the new link coming to this neuron
  local linkTo    -- the new link coming from this neuron

  local weight = link.weight

  local fromNeuron = self:getNeuron(link.from)
  local toNeuron = self:getNeuron(link.to)

  local new_x = (fromNeuron.x+toNeuron.x)/2
  local new_y = (fromNeuron.y+toNeuron.y)/2

  -- checks if innovation has already been created
  local id = innovationList:getID("new_neuron", link.from, link.to)

  if id == -1 then
    -- this is a new innovation
    -- itype, from, to, neuronID, ntype

    local neuronID = innovationList:newNeuronID()
    local neuronInnovation = Innovation("new_neuron", fromNeuron.id, toNeuron.id, neuronID, "hidden")
    local linkFromInnovation = Innovation("new_link", fromNeuron.id, neuronID, -1, "none")
    local linkToInnovation = Innovation("new_link", neuronID, toNeuron.id, -1, "none")

    -- adding innovations to database
    innovationList:push(neuronInnovation)
    local linkFromID = innovationList:push(linkFromInnovation)
    local linkToID = innovationList:push(linkToInnovation)

    neuron = NeuronGene(neuronID, "hidden", false, 1, new_x, new_y) --id, ntype, recurrent, response, x, y

    linkFrom = LinkGene(linkFromID, 1, fromNeuron.id, neuronID, true, false)-- innovation, weight, from, to, enabled, recurrent
    linkTo = LinkGene(linkToID, weight, neuronID, toNeuron.id, true, false)
  else
    -- this innovation was already discovered
    local neuronID = innovationList:getNeuronID("new_neuron", fromNeuron.id, toNeuron.id)
    local lastn = self.NeuronGeneList[#self.NeuronGeneList]

    --[[while self:alreadyHaveThisNeuronID(neuronID) do
      neuronID = innovationList:getNextNeuronID(neuronID, "new_neuron", fromNeuron.id, toNeuron.id)
    end]]

    local linkFromInnovation, linkToInnovation

    if not self:alreadyHaveThisNeuronID(neuronID) then
      linkFromInnovation = innovationList:getID("new_link", fromNeuron.id, neuronID)
      linkToInnovation = innovationList:getID("new_link", neuronID, toNeuron.id)
    else
      return
    end

    --[[if neuronID == -1 then
      -- the last neuron innovation between the selected from and to neurons is already on this genome
      -- so this is actually a "new" innovation  and we have to store it on the database

      neuronID = innovationList:newNeuronID()

      neuronInnovation = Innovation("new_neuron", fromNeuron.id, toNeuron.id, neuronID, "hidden")

      innovationList:push(neuronInnovation)
      linkFromInnovation = innovationList:push(Innovation("new_link", fromNeuron.id, neuronID, -1, "none"))
      linkToInnovation = innovationList:push(Innovation("new_link", neuronID, toNeuron.id, -1, "none"))
    else
      linkFromInnovation = innovationList:getID("new_link", fromNeuron.id, neuronID)
      linkToInnovation = innovationList:getID("new_link", neuronID, toNeuron.id)
    end]]

    neuron = NeuronGene(neuronID, "hidden", false, 1, new_x, new_y) --id, ntype, recurrent, response, x, y
    linkFrom = LinkGene(linkFromInnovation, 1, fromNeuron.id, neuronID, true, false)-- innovation, weight, from, to, enabled, recurrent
    linkTo = LinkGene(linkToInnovation, weight, neuronID, toNeuron.id, true, false)
  end

  self:pushNeuronGene(neuron)
  self:pushLinkGene(linkFrom)
  self:pushLinkGene(linkTo)
end

function Genome:mutatePerturbWeight(parameters)
  for _, link in ipairs(self.LinkGeneList) do
    if random() < parameters.perturbWeight then
      if random() < parameters.replaceWeight then
        link:randomWeight(parameters.weightLimit)
      else
        link.weight = link.weight + (random()*2-1)*parameters.maxPerturbation
      end
    end
  end
end

function Genome:mutatePerturbResponse(parameters)
  for _, neuron in ipairs(self.NeuronGeneList) do
    if random() < parameters.perturbResponse then
      if random() < parameters.alterResponse then
        neuron:alterResponse(parameters.responseLimit)
      else
        neuron.response = neuron.response + (random()*2-1)*parameters.maxResponse
      end
    end
  end
end

function Genome:mutateEnableDisableLink(parameters)
  for _, link in ipairs(self.LinkGeneList) do
    if random() < parameters.enableDisable then
      link.enabled = not link.enabled
    end
  end
end

function Genome:alreadyHaveThisNeuronID(id)
  local lower, upper = 1, #self.NeuronGeneList
  local index

  while lower <= upper do
    index = math.floor((lower+upper)/2)

    if self.NeuronGeneList[index].id > id then
      upper = index-1
    elseif self.NeuronGeneList[index].id < id then
      lower = index+1
    else
      return true
    end
  end

  return false
end

function Genome:getNeuron(id)
  local lower, upper = 1, #self.NeuronGeneList
  local index

  while lower <= upper do
    index = math.floor((lower+upper)/2)

    if self.NeuronGeneList[index].id > id then
      upper = index-1
    elseif self.NeuronGeneList[index].id < id then
      lower = index+1
    else
      return self.NeuronGeneList[index]
    end
  end
end

function Genome:getLink(innovation)
  local lower, upper = 1, #self.LinkGeneList
  local index

  while lower <= upper do
    index = math.floor((lower+upper)/2)

    if self.LinkGeneList[index].innovation > innovation then
      upper = index-1
    elseif self.LinkGeneList[index].innovation < innovation then
      lower = index+1
    else
      return self.LinkGeneList[index]
    end
  end
end

function Genome:getRandomNeuron()
  local index = random(1, #self.NeuronGeneList)

  return self.NeuronGeneList[index]
end

function Genome:getRandomLink()
  local index = random(1, #self.LinkGeneList)

  return self.LinkGeneList[index]
end

function Genome:getRandomOldLink()
  local index = random(1, #self.LinkGeneList-math.floor(math.sqrt(#self.LinkGeneList)))

  return self.LinkGeneList[index]
end

function Genome:getBaseNeuronsAmount()
  -- returns the number of input, bias and otput neurons in this genome
  local count = 0
  for _, neuron in ipairs(self.NeuronGeneList) do
    if neuron.ntype == "input" or neuron.ntype == "bias" or neuron.ntype == "output" then
      count = count + 1
    else
      break
    end
  end

  return count
end

function Genome:pushNeuronGene(neuronGene)
  if #self.NeuronGeneList == 0 then
    table.insert(self.NeuronGeneList, neuronGene); return
  end

  local lower, upper = 1, #self.NeuronGeneList
  local index=0

  while lower <= upper do
    index = math.floor((lower+upper)/2)

    if self.NeuronGeneList[index].id == neuronGene.id then
      -- neuron is already on the list
      return
    elseif self.NeuronGeneList[index].id > neuronGene.id then
      upper = index-1
    elseif self.NeuronGeneList[index].id < neuronGene.id then
      lower = index+1
    end
  end

  if self.NeuronGeneList[index].id < neuronGene.id then
    table.insert(self.NeuronGeneList, index+1, neuronGene)
  else
    table.insert(self.NeuronGeneList, index, neuronGene)
  end
end

function Genome:pushLinkGene(linkGene)
  if #self.LinkGeneList == 0 then
    table.insert(self.LinkGeneList, linkGene); return
  end

  local lower, upper = 1, #self.LinkGeneList
  local index=0

  while lower <= upper do
    index = math.floor((lower+upper)/2)

    if self.LinkGeneList[index].innovation == linkGene.innovation then
      -- neuron is already on the list
      return
    elseif self.LinkGeneList[index].innovation > linkGene.innovation then
      upper = index-1
    elseif self.LinkGeneList[index].innovation < linkGene.innovation then
      lower = index+1
    end
  end

  if self.LinkGeneList[index].innovation < linkGene.innovation then
    table.insert(self.LinkGeneList, index+1, linkGene)
  else
    table.insert(self.LinkGeneList, index, linkGene)
  end
end

function Genome:linkExists(neuron1, neuron2)
  for _, link in ipairs(self.LinkGeneList) do
    if link.from == neuron1.id and link.to == neuron2.id then
      return true
    end
  end

  return false
end

function Genome:linkGeneExists(link)
  local lower, upper = 1, #self.LinkGeneList
  local index

  while lower <= upper do
    index = math.floor((lower+upper)/2)

    if self.LinkGeneList[index].innovation > link.innovation then
      upper = index-1
    elseif self.LinkGeneList[index].innovation < link.innovation then
      lower = index+1
    else
      return true
    end
  end

  return false
end

function Genome:copy()
  local genome = Genome(self.id)

  genome.species = self.species
  genome.fitness = self.fitness
  genome.adjustedFitness = self.adjustedFitness

  for _, neuron in ipairs(self.NeuronGeneList) do
    table.insert(genome.NeuronGeneList, neuron:copy())
  end

  for _, link in ipairs(self.LinkGeneList) do
    table.insert(genome.LinkGeneList, link:copy())
  end

  return genome
end

--------------------------------
--          GENES             --
--------------------------------

-- neuron gene
NeuronGene = {}
NeuronGene.mt = {
  __index = NeuronGene,

  __tostring = function(self)
    return     "ID:\t\t" .. self.id
            .. "\nType:\t\t" .. self.ntype
            .. "\nRecurrent:\t" .. tostring(self.recurrent)
            .. "\nResponse:\t" .. self.response
            .. "\nx:\t\t" .. self.x
            .. "\ny:\t\t" .. self.y
  end,
}

setmetatable(NeuronGene, {
  __call = function(t, id, ntype, recurrent, response, x, y)
    local o = {}
    o.id        = id
    o.ntype     = ntype
    o.recurrent = recurrent
    o.response  = response or 1
    o.x         = x
    o.y         = y

    -- ntype (neuron type) can be: "input", "output", "hidden" and "bias"
    -- response refers to the response of the sigmoid function (p)

    return setmetatable(o, NeuronGene.mt)
  end
})

function NeuronGene:alterResponse(responseLimit)
  self.response = random()*(responseLimit)-responseLimit
end

function NeuronGene:copy()
  return NeuronGene(self.id, self.ntype, self.recurrent, self.response, self.x, self.y)
end

-- link gene
LinkGene = {}
LinkGene.mt = {
  __index = LinkGene,

  __tostring = function(self)
    return  "Innovation:\t" .. self.innovation
          .."\nWeight:\t\t" .. self.weight
          .."\nFrom:\t\t" .. self.from
          .."\nTo:\t\t" .. self.to
          .."\nEnabled:\t" .. tostring(self.enabled)
          .."\nRecurrent:\t" .. tostring(self.recurrent)
  end
}

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

function LinkGene:randomWeight(weightLimit)
  self.weight = random()*(weightLimit*2)-weightLimit
end

function LinkGene:copy()
  return LinkGene(self.innovation, self.weight, self.from, self.to, self.enabled, self.recurrent)
end

--------------------------------
--       NEURAL NETWORK       --
--------------------------------

NeuralNetwork = {}
NeuralNetwork.mt = {
  __index = NeuralNetwork,

  __tostring = function(self)
    return nil
  end
}

setmetatable(NeuralNetwork, {
  __call = function(t, inputs, outputs, bias, neuronList, genome)
    local o = {}

    o.inputs = inputs
    o.outputs = outputs
    o.bias = bias
    o.depth = 1
    o.NeuronList = neuronList or {}
    o.genome = genome

    return setmetatable(o, NeuralNetwork.mt)
  end
})

function NeuralNetwork:forward(inputs, method)
  if type(inputs) ~= "table" then
    error("inputs must be a table", 2)
  end

  if #inputs ~= self.inputs then
    --error("number of inputs do not match those of the neural net (".. self.inputs .. ")", 2)
    error("incorrect number of inputs (expected ".. self.inputs .. "; got ".. #inputs .. ")", 2)
  end

  method = method or "active"

  if method ~= "active" and method ~= "snapshot" then
    error("unknown network update method", 2)
  end

  local outputs = {}
  local flushCount

  if method == "snapshot" then
    flushCount = self.depth

    -- sets the inputs only once when snapshot
    for n = 1, #inputs do
      local neuron = self.NeuronList[n]

      neuron.output = inputs[n]
    end
  else
    flushCount = 1
  end

  if self.NeuronList[#inputs+1].ntype == "bias" then
    self.NeuronList[#inputs+1].output = 1
    bias = 1
  end

  for f = 1, flushCount do
    outputs = {}

    -- setting the new inputs each flush
    if method == "active" then
      for n = 1, #inputs do
        local neuron = self.NeuronList[n]

        neuron.output = inputs[n]
      end
    end

    for index = self.inputs+bias, #self.NeuronList do
      local neuron = self.NeuronList[index]

      local sum = 0
      for _, link in ipairs(neuron.incoming) do
        local weight = link.weight
        local value = self:getNeuron(link.input).output

        sum = sum + value*weight
      end

      neuron.output = sigmoid(sum, neuron.response)

      if neuron.ntype == "output" then
        table.insert(outputs, neuron.output)
      end
    end
  end

  -- clear all neurons if method is snapshot
  if method == "snapshot" then
    self:clear()
  end

  return outputs
end

function NeuralNetwork:draw(width, height, sx, sy)
  self.genome:draw(width, height, sx, sy)
end

function NeuralNetwork:clear()
  for _, neuron in ipairs(self.NeuronList) do
    neuron.output = 0
  end
end

function NeuralNetwork:getNeuron(id)
  local lower, upper = 1, #self.NeuronList-self.outputs
  local index

  --[[if id > self.inputs+1 and id < self.inputs+1+self.outputs+1 then
    for n = #self.NeuronList, 1, -1 do
      if self.NeuronList[n].id == id then
        return self.NeuronList[n]
      end
    end
  end]]

  --[[if id <= self.inputs+self.bias then
    return self.NeuronList[id]
  elseif id > self.inputs + self.bias and id <= self.inputs + self.bias + self.outputs then
    local i = #self.NeuronList - self.outputs + id - self.inputs - self.bias
    return self.NeuronList[i]
  end]]

  for i, neuron in ipairs(self.NeuronList) do
    if neuron.id == id then
      return neuron
    end
  end

  --[[while lower <= upper do
    index = math.floor((lower+upper)/2)

    if self.NeuronList[index].id > id then
      upper = index-1
    elseif self.NeuronList[index].id < id then
      lower = index+1
    else
      return self.NeuronList[index]
    end
  end]]

  --return self:getNeuron(id, true)
end

function NeuralNetwork:setFitness(fitness)
  self.genome.fitness = fitness or 0
end

function NeuralNetwork:incrementFitness(increment)
  self.genome.fitness = self.genome.fitness + (increment or 0)
end

function NeuralNetwork:getFitness()
  return self.genome.fitness
end

Neuron = {}
Neuron.mt = {__index = Neuron}

setmetatable(Neuron, {
  __call = function(t, id, neuronType, response)
    local o = {}

    o.id = id or -1 -- the neuron gene id
    o.incoming = {}
    o.leaving = {}
    o.sum = 0
    o.output = 0
    o.ntype = neuronType
    o.response = response

    return setmetatable(o, Neuron.mt)
  end
})

Link = {}
Link.mt = {__index = Link}

setmetatable(Link, {
  __call = function(t, input, output, weight, recurrent)
    local o = {}

    o.input = input
    o.output = output
    o.weight = weight
    o.recurrent = recurrent

    return setmetatable(o, Link.mt)
  end
})

--------------------------------
--         INNOVATION         --
--------------------------------

Innovation = {}
Innovation.mt = {
  __index = Innovation,

  __tostring = function(self)
    return    "\nType:\t\t" .. self.itype
           .. "\nFrom:\t\t" .. self.from
           .. "\nTo:\t\t" .. self.to
           .. "\nNeuron ID:\t" .. self.neuronID
           .. "\nNeuron type:\t" .. self.ntype
  end
}

setmetatable(Innovation, {
  __call = function(t, itype, from, to, neuronID, ntype)
    local o = {}

    o.itype     = itype
    o.from      = from
    o.to        = to
    o.neuronID  = neuronID
    o.ntype     = ntype

    return setmetatable(o, Innovation.mt)
  end
})

--------------------------------
--      INNOVATION LIST       --
--------------------------------

InnovationList = {}
InnovationList.mt = {
  __index = InnovationList,

  __tostring = function(self)
    local str = ""

    for i, innovation in ipairs(self) do
      str = str .. tostring(innovation)

      if i<#self then str = str .. "\n\n" end
    end

    return str
  end
}

setmetatable(InnovationList, {
  __call = function(t)
    return setmetatable({}, InnovationList.mt)
  end
})

function InnovationList:getID(itype, from, to)
  for id, i in ipairs(self) do
    if i.itype==itype and i.from==from and i.to==to then
      return id
    end
  end

  return -1
end

function InnovationList:getNeuronID(itype, from, to)
  local id = self:getID(itype, from, to)

  if id > 0 then
    return (self[id]).neuronID
  end

  return -1
end

function InnovationList:newNeuronID()
  for n = #self, 1, -1 do
    local innovation = self[n]

    if innovation.itype == "new_neuron" then
      return innovation.neuronID+1
    end
  end

  return 1
end

function InnovationList:getNextNeuronID(neuronID, itype, from, to) -- remove this later?
  local index = self:getNeuronID(id)
  if index==-1 then return -1 end

  for n=index, #self do
    local i = self[n]

    if i.itype==itype and i.from==from and i.to==to then
      return (self[n]).neuronID
    end
  end

  return -1
end

function InnovationList:push(innovation)
  table.insert(self, innovation)
  return #self
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
  o.bestFitnessSoFar = 0
  o.averageFitness = 0
  o.generationsNoImprovement = 0
  o.age = 0
  o.spawnAmount = 0

  return setmetatable(o, Species.mt)
end})

function Species:adjustFitnesses(parameters)
  -- "this method boosts the fitnesses of the young, penalizes the
  --  fitnesses of the old and then performs fitness sharing over
  --  all the members of the species"
  for _, genome in ipairs(self.genomes) do
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

function Species:tournamentSelection(tournamentSize, shift)
  local index
  local best

  for n = 1, tournamentSize do
    index = random(n + (shift or 0), #self.genomes)

    if n + (shift or 0) > #self.genomes then
      if best == nil then
        return
      else
        return best
      end
    end

    if not best then
      best = n
    else
      if self.genomes[index].fitness > self.genomes[best].fitness then
        best = n
      end
    end

    self.genomes[n], self.genomes[index] = self.genomes[index], self.genomes[n]
  end

  return best
end

function Species:generateOffspring(id, parameters, innovationList)
  local genome1, genome2, offspring

  genome1 = self:tournamentSelection(1)
  self.genomes[1], self.genomes[genome1] = self.genomes[genome1], self.genomes[1]
  genome2 = self:tournamentSelection(1, 1)

  genome1 = self.genomes[1]
  genome2 = self.genomes[genome2]

  if random() < parameters.crossoverRate and genome2 then
    offspring = Genome.crossover(-1, genome1, genome2)
  else
    -- copying genome 1
    offspring = genome1:copy()
  end

  offspring.id = id or -1
  offspring.fitness = 0
  offspring.adjustedFitness = 0

  offspring:mutate(parameters, innovationList)

  return offspring
end

function Species:getBestGenome()
  local bestGenome = self.genomes[1]

  for _, genome in ipairs(self.genomes) do
    if genome.fitness > bestGenome.fitness then
      bestGenome = genome
    end
  end

  return bestGenome
end

------------------------
--         POOL       --
------------------------

Pool = {}
Pool.mt = {
  __index = Pool,

  __tostring = function(self)
    return    "Number of species:\t" .. #self.species
           .. "\nNumber of inputs:\t" .. (self.numberOfInputs or 0)
           .. "\nNumber of outputs:\t" .. (self.numberOfOutputs or 0)
  end,
}

setmetatable(Pool, {
  __call = function(t, size, inputs, outputs, noBias)
    local o = {}

    o.size = size
    o.inputs = inputs
    o.outputs = outputs
    o.noBias = noBias or false
    o.species = {}
    o.neuralNetworks = {}
    o.lastBestGenome = nil
    o.topFitness = 0
    o.generation = -1
    o.genomeIdCounter = 0
    o.speciesIdCounter = 0
    o.innovationList = InnovationList()
    o.parameters = {}

    for k, v in pairs(LuaNEAT.DefaultParameters) do
      o.parameters[k] = v
    end

    -- initialize pool with inputs, bias and outputs neuron innovations
    for n=1,inputs do
      local neuronID = o.innovationList:newNeuronID()
      local innovation = Innovation("new_neuron", -1, -1, neuronID, "input")
      o.innovationList:push(innovation)
    end

    -- bias neuron
    if not noBias then
      local neuronID = o.innovationList:newNeuronID()
      local biasInnovation = Innovation("new_neuron", -1, -1, neuronID, "bias")
      o.innovationList:push(biasInnovation)
    end

    for n=1,outputs do
      local neuronID = o.innovationList:newNeuronID()
      local innovation = Innovation("new_neuron", -1, -1, neuronID, "output")
      o.innovationList:push(innovation)
    end

    return setmetatable(o, Pool.mt)
  end
})

function Pool:initialize()
  for n=1, self.size do
    local id = self:getNextGenomeID()
    local genome = Genome.basicGenome(id, self.inputs, self.outputs, self.noBias)

    for n = 1, math.random(1, self.inputs*self.outputs) do
      genome:mutateAddLink(self.parameters, self.innovationList, true)
    end

    table.insert(self.neuralNetworks, genome:buildNeuralNetwork())

    self:pushAndSpeciateGenome(genome)
  end

  self.generation = 0
end

function Pool:getBestPerformer()
  local best

  for _, net in ipairs(self.neuralNetworks) do
    if not best then
      best = net
    else
      if net.genome.fitness > best.genome.fitness then
        best = net
      end
    end
  end

  return best
end

function Pool:getLastBestPerformer()
  return self.lastBestGenome:buildNeuralNetwork()
end

function Pool:getBestFitness()
  local best = self:getBestPerformer()

  if best then
    return best.genome.fitness
  end
end

function Pool:getNeuralNetworks()
  return self.neuralNetworks
end

function Pool:getSpeciesAmount()
  return #self.species
end

function Pool:getRandomSpecies()
  local index = random(1, #self.species)

  return self.species[index]
end

function Pool:getBestGenome()
  local bestGenome = self.species[1].genomes[1]

  for _, species in ipairs(self.species) do
    for __, genome in ipairs(species.genomes) do
      if genome.fitness > bestGenome.fitness then
        bestGenome = genome
      end
    end
  end

  return bestGenome
end

function Pool:getNextGenomeID()
  self.genomeIdCounter = self.genomeIdCounter + 1

  return self.genomeIdCounter
end

function Pool:getNextSpeciesID()
  self.speciesIdCounter = self.speciesIdCounter + 1

  return self.speciesIdCounter
end

function Pool:pushAndSpeciateGenome(genome)
  for _, species in ipairs(self.species) do
    local leader = species.leader
    local distance = Genome.calculateCompatibilityDistance(genome, leader, self.parameters)

    if distance <= self.parameters.sameSpeciesThreshold then
      genome.species = species.id
      table.insert(species.genomes, genome); return
    end
  end

  -- no species is compatible with genome or there are no species yet
  local id = self:getNextSpeciesID()
  local species = Species(id, genome)

  table.insert(species.genomes, genome)
  table.insert(self.species, species)
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
  for _, species in ipairs(self.species) do
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

function Pool:removeStaleSpecies()
  local survived = {}

  for _, species in ipairs(self.species) do
    local bestGenome = species:getBestGenome()

    if bestGenome.fitness > species.bestFitnessSoFar then
      species.bestFitnessSoFar = bestGenome.fitness
      species.generationsNoImprovement = 0
    else
      species.generationsNoImprovement = species.generationsNoImprovement + 1
    end

    if species.generationsNoImprovement < self.parameters.generationsNoImprovement or bestGenome.fitness >= self.topFitness then
      table.insert(survived, species)
    end
  end

  if #survived == 0 then return; end

  self.species = survived
end

function Pool:removeWeakSpecies()
  local survived = {}

  for _, species in ipairs(self.species) do
    if species.spawnAmount >= 1 then
      table.insert(survived, species)
    end
  end

  if #survived == 0 then return; end

  self.species = survived
end

function Pool:getTopFitness()
  return self.topFitness
end

function Pool:getAverageFitness()
  local sum = 0
  local count = 0

  for _, species in ipairs(self.species) do
    for __, genome in ipairs(species.genomes) do
      sum = sum + genome.fitness
      count = count + 1
    end
  end

  return sum/count
end

function Pool:getAverageAdjustedFitness()
  local sum = 0
  local count = 0
  for _, species in ipairs(self.species) do
    for __, genome in ipairs(species.genomes) do
      count = count + 1
      sum = sum + genome.adjustedFitness
    end
  end

  return sum/count
end

function Pool:nextGeneration()
  if self.generation == -1 then
    error("pool needs to be initialized before computing next generation", 2)
  end

  -- derezzing any networks created during last generation
  self.neuralNetworks = {}

  local zero = true

  for _, species in ipairs(self.species) do
    species:adjustFitnesses(self.parameters)

    for __, genome in ipairs(species.genomes) do
      if genome.fitness ~= 0 then
        zero = false
      end
    end
  end

  print("Generation ".. self.generation)
  print("Top fitness is ".. self.topFitness)
  print(#self.species .. " species")

  if zero then
    print("total fitness is 0; reseting pool\n")

    self.species = {}

    self:initialize()

    return
  end

  local bestGenome = self:getBestGenome()
  self.lastBestGenome = bestGenome

  if bestGenome.fitness > self.topFitness then
    self.topFitness = bestGenome.fitness
  end; sp = #self.species

  self:calculateSpawnLevels()

  self:removeWeakSpecies(); if sp - #self.species > 0 then print(sp - #self.species .. " weak species removed") end; sp = #self.species
  self:removeStaleSpecies(); if sp - #self.species > 0 then print(sp - #self.species .. " stale species removed") end

  self:cullSpecies()

  sp = #self.species

  -- generating offsprings
  local spawn = {}

  for _, species in ipairs(self.species) do
    local toSpawn = species.spawnAmount
    local offspring

    for n = 1, math.floor(toSpawn)-1 do
      table.insert(spawn, species:generateOffspring(self:getNextGenomeID(), self.parameters, self.innovationList))
    end
  end

  while (#self.species + #spawn) < self.size do
    local species = self:getRandomSpecies()

    table.insert(spawn, species:generateOffspring(self:getNextGenomeID(), self.parameters, self.innovationList))
  end

  if #self.species-sp > 0 then
    print(#self.species .. " new species spawned")
  end

  self:cullSpecies(true)

  for _, species in ipairs(self.species) do
    species.leader = species.genomes[1]
    species.leader.fitness = 0
    species.leader.adjustedFitness = 0
  end

  -- inserting offsprings into species
  for _, offspring in ipairs(spawn) do
    self:pushAndSpeciateGenome(offspring)
  end

  -- creating networks
  for _, species in ipairs(self.species) do
    for __, genome in ipairs(species.genomes) do
      --local net = genome:buildNeuralNetwork()
      table.insert(self.neuralNetworks, genome:buildNeuralNetwork())
      --table.insert(self.neuralNetworks, net)

      --if self.generation+1 >= 100 then
        --for _, neuron in ipairs(net.NeuronList) do
          --print(neuron.id, neuron.y, neuron.ntype)
        --end
        --error("terminated")
        --while true do end
      --end
    end
  end

  print("\n")

  self.generation = self.generation + 1
end

--------------------------------
--      PUBLIC FUNCTIONS      --
--------------------------------

function LuaNEAT.newPool(size, inputs, outputs, noBias)
  -- error handling: size, inputs and outputs needs to be integers greater than 0

  -- handling size
  if type(size)~="number" then
    error("bad argument: number expected, got ".. type(size), 2)
  end
  if size<=0 then
    error("bad argument: size needs to be greater than 0", 2)
  end
  if math.floor(size)~=size then
    error("bad argument: size needs to be a whole number", 2)
  end

  -- handling inputs
  if type(inputs)~="number" then
    error("bad argument: number expected, got ".. type(inputs), 2)
  end
  if inputs<=0 then
    error("bad argument: inputs needs to be greater than 0", 2)
  end
  if math.floor(inputs)~=inputs then
    error("bad argument: inputs needs to be a whole number", 2)
  end

  -- handling outputs
  if type(outputs)~="number" then
    error("bad argument: number expected, got ".. type(outputs), 2)
  end
  if outputs<=0 then
    error("bad argument: outputs needs to be a whole number", 2)
  end
  if math.floor(outputs)~=outputs then
    error("bad argument: outputs needs to be a whole number", 2)
  end

  noBias = (noBias == true)

  return Pool(size, inputs, outputs, noBias)
end

function LuaNEAT.version()
  return LuaNEAT._VERSION
end





-- public
local size = 250
local inputs = 6
local outputs = 4
local noBias = false

pool = LuaNEAT.newPool(size, inputs, outputs, noBias)
pool:initialize()

--functions intended for public use:

--Pool:getNeuralNetworks()
--Pool:getBestPerformer()
--Pool:getLastBestPerformer()
--Pool:getBestFitness()
--Pool:getAverageFitness()
--Pool:getTopFitness()
--Pool:getSpeciesAmount()
--NeuralNetwork:setFitness()
--NeuralNetwork:incrementFitness()
--NeuralNetwork:getFitness()
--NeuralNetwork:drawGenome()

return LuaNEAT
