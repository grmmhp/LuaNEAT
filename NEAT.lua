local LuaNEAT = {
  _VERSION = "LuaNEAT Alpha Development",
  _DESCRIPTION = "NEAT module for Lua",
  _URL = "https://github.com/grmmhp/LuaNEAT",
  _LICENSE = [[
    MIT License

    Copyright (c) 2018 Gabriel Mesquita

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
  ]]
}

-- TODO:
-- improve Genome:drawNeuralNetwork()
-- code Genome:buildNeuralNetwork()
-- code Genome:mutate()
-- code Genome:countDisjointExcessGenes()
-- expand usage of binary insert
-- test mutate alter response
-- user needs to set random seed for LuaNEAT to work
-- remove randomseed, math.random and oldrandom

-------------------------------------------------------------------------------------------------
-- These functions are used for testing and debugging and will be removed on the first release --
-------------------------------------------------------------------------------------------------

math.randomseed(os.time())

-- prints only when DEBUG is true
local DEBUG = true

local oldprint = print
local print = function(str)
  if DEBUG then oldprint(str) end
end

-- Lua's math.random() is glitchy on Windows, this will fix:
local oldrandom = math.random
math.random = function(_min, _max)
  local out
  if _min == nil then
    -- no parameters, generate random number in [0, 1]
    for n=1, 10 do
      out = oldrandom()
    end
  else
    -- _min is now _max, generate random number in [0, _min]
    if _max == nil then
      for n=1, 10 do
        out = oldrandom(_min)
      end
    else
      for n=1, 10 do
        out = oldrandom(_min, _max)
      end
    end
  end

  return out
end

------------
-- MODULE --
------------

-- forward declaration of classes
local Genome, NeuronGene, LinkGene, Innovation, InnovationList, Population

local function sigmoid(x, p)
  local e = 2.71828182846
  return 1/(1+e^(-x*p))
end

-- rates prototype:
local ratesPrototype = {
  loopedLink = .1,
  perturbWeight = .1,
  maxPerturbation = .1,
  replaceWeight = .1,
  perturbResponse = .1,
  alterResponse = .1,
  maxResponse = .1,
}

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
    o.species = "none"
    o.fitness = 0
    o.NeuronGeneList = NeuronGeneList
    o.LinkGeneList = LinkGeneList

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
end

function Genome:drawNeuralNetwork(width, height, sx, sy)
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
      love.graphics.setColor(1, 0, 0, 1)
    else
      love.graphics.setColor(0, 1, 0, 1)
    end

    love.graphics.circle("fill", x, y, radius)
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.print(neuron.id, x-radius/2, y-radius/2)
  end
end

function Genome.crossover(genome1, genome2)
  -- genome1 is always the more fit genome
  -- "In composing the offspring, genes are randomly chosen from either parent at matching genes,
  -- whereas all excess or disjoint genes are always included from the more fit parent"
  local offspring
  local neuronGeneList = {}
  local linkGeneList = {}

  local function binaryInsert(neuron)
    if #neuronGeneList == 0 then return
      table.insert(neuronGeneList, neuron)
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

  print("INITIAZING CROSSOVER\nGenome IDs are ".. genome1.id .. " and ".. genome2.id)

  if genome2.fitness > genome1.fitness then
    -- swap the genomes
    genome1, genome2 = genome2, genome1
  elseif genome1.fitness == genome2.fitness then
    -- equal fitness, choose one at random
    if math.random() < .5 then
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

      if math.random() < .5 then
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

  offspring = Genome(-1, neuronGeneList, linkGeneList)

  print("FINISHED CROSSOVER")

  return offspring
end

function Genome:mutate(rates, innovationList)
  -- mutations in this implementation:

  -- add link
  -- add neuron
  -- perturb weight
  -- alter response curve
end

function Genome:mutateAddLink(rates, innovationList)
  -- adds a forward, recurrent or looped link

  print("INITIATING LINK MUTATION\nGenome ID is ".. self.id .."\n")

  local findLinkAttempts = 20

  local neuron1
  local neuron2
  local recurrent = false

  if math.random() < rates.loopedLink then
    -- tries to find a hidden or output neuron that does not already have a loopback
    -- looped link is going to be added
    for n=1, findLinkAttempts do
      print("#".. n .. " ATTEMPT TO FIND LOOPED LINK")
      neuron1 = self:getRandomNeuron()

      if  (neuron1.ntype ~= "input")
      and (neuron1.ntype ~= "bias")
      and (not neuron1.recurrent) then
        neuron2 = neuron1
        neuron1.recurrent = true

        recurrent = true

        print("neuron found: ".. neuron1.id)

        break
      end
    end
  else
    -- tries to find two unconnected neurons
    for n=1, findLinkAttempts do
      print("#".. n .. " ATTEMPT TO FIND NORMAL LINK")
      neuron1 = self:getRandomNeuron()
      neuron2 = self:getRandomNeuron()

      if (not self:linkExists(neuron1, neuron2)) and (neuron1.id ~= neuron2.id) and (neuron2.ntype~="input") and (neuron2.ntype~="bias") then
        print("neuron 1 found: ".. neuron1.id)
        print("neuron 2 found: ".. neuron2.id)
        break
      else
        neuron1 = nil
        neuron2 = nil
      end
    end
  end

  -- could not find neurons
  if neuron1==nil or neuron2==nil then print("COULD NOT FIND NEURONS"); return end

  -- checking innovation
  local innovationID = innovationList:getID("new_link", neuron1.id, neuron2.id)

  -- checking if link is recurrent
  if neuron1.y < neuron2.y then
    recurrent = true
  end

  if innovationID == -1 then
    -- innovation wasnt discovered yet
    local innovation = Innovation("new_link", neuron1.id, neuron2.id, -1, "none")
    -- id, itype, from, to, neuronID, ntype
    innovationID = innovationList:push(innovation)
  else
    -- innovation was already discovered
  end

  --innovation, weight, from, to, enabled, recurrent
  local linkGene = LinkGene(innovationID, 0, neuron1.id, neuron2.id, true, recurrent)
  linkGene:randomWeight()

  self:pushLinkGene(linkGene)

  print("FINISHED MUTATING LINK")
end

function Genome:mutateAddNode(rates, innovationList)
  print("INITIATING NODE MUTATION\nGenome ID is ".. self.id .. "\n")
  local link
  local sizeThreshold = self:getBaseNeuronsAmount()+5
  local findLinkAttempts = 20

  if #self.LinkGeneList == 0 then print("GENOME HAS NO LINKS"); return end

  if #self.LinkGeneList < sizeThreshold then
    -- genome is too small; will choose an old link
    print("genome is too small; will choose older links")
    for n=1, findLinkAttempts do
      print("#".. n .. " attempt to find link")
      link = self:getRandomOldLink()

      local fromNeuron = self:getNeuron(link.from)

      if (link.enabled) and (not link.recurrent) and (fromNeuron.ntype~="bias") then
        print("found link ".. link.innovation)
        break
      else
        link = nil
      end
    end
  else
    -- genome is of sufficient size
    print("genome is of sufficient size; will choose any link")
    while true do
      link = self:getRandomLink()

      local fromNeuron = self:getNeuron(link.from)

      if (link.enabled) and (not link.recurrent) and (fromNeuron.ntype~="bias") then
        break
      end
    end
  end

  if link == nil then print("COULD NOT FIND LINK"); return end

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
    print("new innovation found")

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
    print("innovation already discovered; will get IDs")

    local neuronID = innovationList:getNeuronID("new_neuron", fromNeuron.id, toNeuron.id)

    while self:alreadyHaveThisNeuronID(neuronID) do
      neuronID = innovationList:getNextNeuronID(neuronID, "new_neuron", fromNeuron.id, toNeuron.id)
    end

    local neuronInnovation = innovationList:getNeuronInnovationID("new_neuron", fromNeuron.id, toNeuron.id, neuronID)
    local linkFromInnovation = innovationList:getID("new_link", fromNeuron.id, neuronID)
    local linkToInnovation = innovationList:getID("new_link", neuronID, toNeuron.id)

    neuron = NeuronGene(neuronID, "hidden", false, 1, new_x, new_y) --id, ntype, recurrent, response, x, y

    linkFrom = LinkGene(linkFromInnovation, 1, fromNeuron.id, neuronID, true, false)-- innovation, weight, from, to, enabled, recurrent
    linkTo = LinkGene(linkToInnovation, weight, neuronID, toNeuron.id, true, false)
  end

  print(linkFrom.innovation ..", ".. linkFrom.weight)
  print(linkTo.innovation ..", ".. linkTo.weight)

  self:pushNeuronGene(neuron)
  self:pushLinkGene(linkFrom)
  self:pushLinkGene(linkTo)

  print("FINISHED ADD NODE MUTATION")
end

function Genome:mutatePerturbWeight(rates)
  for _, link in ipairs(self.LinkGeneList) do
    if math.random() < rates.perturbWeight then
      if math.random() < rates.replaceWeight then
        link:randomWeight()
      else
        link.weight = link.weight + math.random()*rates.maxPerturbation
      end
    end
  end
end

function Genome:mutatePerturbResponse(rates)
  for _, neuron in ipairs(self.NeuronGeneList) do
    if math.random() < rates.perturbResponse then
      if math.random() < rates.alterResponse then
        neuron:alterResponse()
      else
        neuron.response = neuron.response + math.random()*rates.maxResponse
      end
    end
  end
end

function Genome:alreadyHaveThisNeuronID(id)
  for _, neuron in ipairs(self.NeuronGeneList) do
    if neuron.id == id then
      return true
    end
  end

  return false
end

function Genome:getNeuron(id)
  for _, neuron in ipairs(self.NeuronGeneList) do
    if neuron.id == id then
      return neuron
    end
  end
end

function Genome:getLink(innovation)
  for _, link in ipairs(self.LinkGeneList) do
    if link.innovation == innovation then
      return link
    end
  end
end

function Genome:getRandomNeuron()
  local index = math.random(1, #self.NeuronGeneList)

  return self.NeuronGeneList[index]
end

function Genome:getRandomLink()
  local index = math.random(1, #self.LinkGeneList)

  return self.LinkGeneList[index]
end

function Genome:getRandomOldLink()
  local index = math.random(1, #self.LinkGeneList-math.floor(math.sqrt(#self.LinkGeneList)))

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
  table.insert(self.NeuronGeneList, neuronGene)
end

function Genome:pushLinkGene(linkGene)
  table.insert(self.LinkGeneList, linkGene)
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
  for _, gene in ipairs(self.LinkGeneList) do
    if gene.innovation == link.innovation then
      return true
    end
  end

  return false
end

function Genome:countDisjointExcessGenes(genome1, genome2)
  -- still to code; will be used in speciation
end

function Genome:getFitness()
  return self.fitness
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
    o.response  = response or math.random()*20-10
    o.x         = x
    o.y         = y

    -- ntype (neuron type) can be: "input", "output", "hidden" and "bias"
    -- response refers to the response of the sigmoid function (p)

    return setmetatable(o, NeuronGene.mt)
  end
})

function NeuronGene:copy()
  return NeuronGene(self.id, self.ntype, self.recurrent, self.response, self.x, self.y)
end

function NeuronGene:alterResponse()
  self.response = math.random()*20-10
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

function LinkGene:randomWeight()
  self.weight = math.random()*4-2
end

function LinkGene:copy()
  return LinkGene(self.innovation, self.weight, self.from, self.to, self.enabled, self.recurrent)
end

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

function InnovationList:getNeuronInnovationID(itype, from, to, neuronID)
  for id, i in ipairs(self) do
    if i.itype==itype and i.from==from and i.to==to and i.neuronID==neuronID then
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
  for n=#self, 1, -1 do
    local innovation = self[n]

    if innovation.itype == "new_neuron" then
      return innovation.neuronID+1
    end
  end

  return 1
end

function InnovationList:getNextNeuronID(id, itype, from, to)
  local index = self:getNeuronID(id)
  if index==-1 then return self:newNeuronID() end

  for n=index, #self do
    local i = self[n]

    if i.itype==itype and i.from==from and i.to==to then
      return (self[n]).neuronID
    end
  end

  return self:newNeuronID()
end

function InnovationList:push(innovation)
  table.insert(self, innovation)
  return #self
end

function InnovationList:next()
  return #self+1
end

--------------------------------
--         POPULATION         --
--------------------------------

Population = {}
Population.mt = {
  __index = Population,

  __tostring = function(self)
    return    "Number of genomes:\t" .. #self.genomes
           .. "\nNumber of inputs:\t" .. (self.numberOfInputs or 0)
           .. "\nNumber of outputs:\t" .. (self.numberOfOutputs or 0)
  end,
}

setmetatable(Population, {
  __call = function(t, size, inputs, outputs, noBias)
    local o = {}

    o.size = size
    o.inputs = inputs
    o.outputs = outputs
    o.noBias = noBias or false
    o.genomes = {}
    o.genomeIdCounter = 0
    o.innovationList = InnovationList()
    o.mutationChances = {
      addLink           = 1,
      addLoopedLink     = 1,
      addRecurrentLink  = 1,
      addNode           = 1,
    }

    -- initialize innovations with inputs and outputs neurons
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

    return setmetatable(o, Population.mt)
  end
})

function Population:getNextGenomeID()
  self.genomeIdCounter = self.genomeIdCounter + 1

  return self.genomeIdCounter
end

--------------------------------
--      PUBLIC FUNCTIONS      --
--------------------------------

function LuaNEAT.newPopulation(size, inputs, outputs, noBias)
  -- error handling: size, inputs and outputs needs to be integers and be greater than 0

  -- handling size
  if type(size)~="number" then
    error("number expected, got ".. type(size), 2)
  end
  if size<=0 then
    error("size needs to be greater than 0", 2)
  end
  if math.floor(size)~=size then
    error("size needs to be a whole number", 2)
  end

  -- handling inputs
  if type(inputs)~="number" then
    error("number expected, got ".. type(inputs), 2)
  end
  if inputs<=0 then
    error("inputs needs to be greater than 0", 2)
  end
  if math.floor(inputs)~=inputs then
    error("inputs needs to be a whole number", 2)
  end

  -- handling outputs
  if type(outputs)~="number" then
    error("number expected, got ".. type(outputs), 2)
  end
  if outputs<=0 then
    error("outputs needs to be a whole number", 2)
  end
  if math.floor(outputs)~=outputs then
    error("outputs needs to be a whole number", 2)
  end

  return Population(size, inputs, outputs, noBias)
end





-- public
size = 200
inputs = 3
outputs = 1
noBias = true

population = LuaNEAT.newPopulation(size, inputs, outputs, noBias)
--print(tostring(population.innovationList))

testRates = {
  loopedLink = .1,
  perturbWeight = 1,
  maxPerturbation = .1,
  replaceWeight = 1,
  perturbResponse = .1,
  alterResponse = .1,
  maxResponse = .1,
}


genome = Genome.basicGenome(1, inputs, outputs, noBias)
genome2 = Genome.basicGenome(2, inputs, outputs, noBias)

genome.fitness = 50
genome2.fitness = 10

num_mutate=10
for n=1, num_mutate do
  if math.random() < .4 then
    genome:mutateAddLink(testRates, population.innovationList);print("\n")
  end

  if math.random() < .3 then
    genome:mutateAddNode(testRates, population.innovationList);print("\n")
  end
end
for n=1, num_mutate do
  if math.random() < .4 then
    genome2:mutateAddLink(testRates, population.innovationList);print("\n")
  end

  if math.random() < .3 then
    genome2:mutateAddNode(testRates, population.innovationList);print("\n")
  end
end

offspring = Genome.crossover(genome2, genome)

print("GENOME\n")
for _, neuron in ipairs(genome.NeuronGeneList) do
  io.write(neuron.id .. " ")
end
for _, link in ipairs(genome.LinkGeneList) do
  --print(tostring(link))
  --print("\n")
end

print("\n\n\nGENOME2\n")

for _, neuron in ipairs(genome2.NeuronGeneList) do
  io.write(neuron.id .. " ")
end
for _, link in ipairs(genome2.LinkGeneList) do
  --print(tostring(link))
  --print("\n")
end

print("\n\n\nOFFSPRING\n")

for _, neuron in ipairs(offspring.NeuronGeneList) do
  io.write(neuron.id .. " ")
end
for _, link in ipairs(offspring.LinkGeneList) do
  --print(tostring(link))
  --print("\n")
end







return LuaNEAT
