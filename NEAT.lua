math.randomseed(os.time())

-- prints texts only when DEBUG is true
local DEBUG = true

local oldprint = print
print = function(str)
  if DEBUG then oldprint(str) end
end

-- Lua's math.random() is glitchy, this will fix:
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


local LuaNEAT = {}

-- TODO:
-- code Genome:buildNeuralNetwork()
-- code Genome:mutate()
-- user needs to set random seed for LuaNEAT to work
-- test mutate add link
-- remove randomseed and randomfix

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

-- forward declaration of classes
local NeuronGene, LinkGene, Genome, Innovation, InnovationList, Population

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

function Genome:mutate(rates)
  -- mutations in this implementation:

  -- add link
  -- add neuron
  -- perturb weight
  -- alter response curve
end

function Genome:mutateAddLink(rates, innovationList)
  -- adds a forward, recurrent or looped link

  print("Initiating link mutation\nGenome ID is ".. self.id .."\n")

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

function Genome:mutateAddNode(rates)
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

function Genome:getRandomNeuron()
  local index = math.random(1, #self.NeuronGeneList)

  return self.NeuronGeneList[index]
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

function NeuronGene:copy()
  return NeuronGene(self.id, self.ntype, self.recurrent)
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

    --o.id        = id
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

function InnovationList:push(innovation)
  --innovation.id = self:next()
  table.insert(self, innovation)
  return #self
end

function InnovationList:get(i)
  return self[i]
end

function InnovationList:next()
  return #self+1
end

function InnovationList:nextNeuronID()
  for n=#self, 1, -1 do
    local innovation = self:get(n)

    if innovation.itype == "new_neuron" then
      return innovation.neuronID+1
    end
  end

  return 1
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
    o.innovationList = InnovationList()
    o.mutationChances = {
      addLink           = 1,
      addLoopedLink     = 1,
      addRecurrentLink  = 1,
      addNeuron         = 1,
    }

    -- initialize innovations with inputs and outputs neurons
    for n=1,inputs do
      local neuronID = o.innovationList:nextNeuronID()
      local innovation = Innovation("new_neuron", -1, -1, neuronID, "input")
      o.innovationList:push(innovation)
    end

    -- bias neuron
    if not noBias then
      local neuronID = o.innovationList:nextNeuronID()
      local biasInnovation = Innovation("new_neuron", -1, -1, neuronID, "bias")
      o.innovationList:push(biasInnovation)
    end

    for n=1,outputs do
      local neuronID = o.innovationList:nextNeuronID()
      local innovation = Innovation("new_neuron", -1, -1, neuronID, "output")
      o.innovationList:push(innovation)
    end

    return setmetatable(o, Population.mt)
  end
})

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
inputs = 2
outputs = 5
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
--link = LinkGene(5, 0.141592653589, 1, 4, true, false)
table.insert(genome.LinkGeneList, link)
--(t, innovation, weight, from, to, enabled, recurrent)


for _,neuron in ipairs(genome.NeuronGeneList) do
  print(tostring(neuron).."\n")
end
print("\n\nbefore")
for _,link in ipairs(genome.LinkGeneList) do
  print(tostring(link).."\n")
end
print("DONE PRINTING LINKS\n")

genome:mutateAddLink(testRates, population.innovationList);print("\n")
genome:mutateAddLink(testRates, population.innovationList);print("\n")
genome:mutateAddLink(testRates, population.innovationList);print("\n")
genome:mutateAddLink(testRates, population.innovationList)

print("\n\nafter")
for _,link in ipairs(genome.LinkGeneList) do
  print(tostring(link).."\n")
end
print("DONE PRINTING LINKS\n")
genome:mutatePerturbWeight(testRates)
print("\n\nafter")
for _,link in ipairs(genome.LinkGeneList) do
  print(tostring(link).."\n")
end
print("DONE PRINTING LINKS\n")







return LuaNEAT
