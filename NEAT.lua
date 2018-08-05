local LuaNEAT = {}

----------------------------TODO----------------------------
-- assign x and y positions on Genome.basicGenome()
-- Genome:buildNeuralNetwork()
--
------------------------------------------------------------



--------------------------------
--          GENES             --
--------------------------------

-- neuron gene
local NeuronGene = {}
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

    return setmetatable(o, NeuronGene.mt)
  end
})

function NeuronGene:copy()
end





-- link gene
local LinkGene = {}
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

function LinkGene:copy()
  return LinkGene(self.innovation, self.weight, self.from, self.to, self.enabled, self.recurrent)
end

--------------------------------
--         MUTATIONS          --
--------------------------------

local function mutateAddNode()
  -- loops through all links
end

local function mutateAddLink()

end

--------------------------------
--           GENOME           --
--------------------------------

local Genome = {}
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

function Genome.basicGenome(id, inputs, outputs)
  local neuronList = {}
  local linkList = {}

  for n=1,inputs do
    -- puts the input neurons evenly spaced on the grid
    local dx = 1/(inputs+1)
    local neuron = NeuronGene(n, "input", false, 1, n*dx, 1)
    table.insert(neuronList, neuron)
  end

  for n=1,outputs do
    -- puts the output neurons evenly spaced on the grid
    local dx = 1/(outputs+1)
    local neuron = NeuronGene(inputs+n, "output", false, 1, n*dx, 0)
    table.insert(neuronList, neuron)
  end

  return Genome(id, neuronList, linkList)
end

function Genome:buildNeuralNetwork()
end

function Genome:alreadyHaveThisNeuronID(id)
  for _, neuron in ipairs(self.NeuronGeneList) do
    if neuron.id == id then
      return true
    end
  end

  return false
end

--------------------------------
--        INNOVATIONS         --
--------------------------------

local Innovation = {}
Innovation.mt = {
  __index = Innovation,

  __tostring = function(self)
    return    "ID:\t\t" .. self.id
           .. "\nType:\t\t" .. self.itype
           .. "\nFrom:\t\t" .. self.from
           .. "\nTo:\t\t" .. self.to
           .. "\nNeuron ID:\t" .. self.neuronID
           .. "\nNeuron type:\t" .. self.ntype
  end
}

setmetatable(Innovation, {
  __call = function(t, id, itype, from, to, neuronID, ntype)
    local o = {}
    o.id        = id
    o.itype     = itype
    o.from      = from
    o.to        = to
    o.neuronID  = neuronID
    o.ntype     = ntype

    return setmetatable(o, Innovation.mt)
  end
})

--------------------------------
--         POPULATION         --
--------------------------------

local Population = {}
Population.mt = {
  __index = Population,

  __tostring = function(self)
    return    "Number of genomes:\t" .. #self.genomes
           .. "\nNumber of inputs:\t" .. (self.numberOfInputs or 0)
           .. "\nNumber of outputs:\t" .. (self.numberOfOutputs or 0)
  end,
}

setmetatable(Population, {
  __call = function(t, size, inputs, outputs)
    local o = {}

    o.size = size
    o.inputs = inputs
    o.outputs = outputs
    o.genomes = {}
    o.innovations = {}

    -- initialize innovations with inputs and outputs neurons

    for n=1,inputs do
      local innovation = Innovation(n, "new_neuron", -1, -1, n, "input")
      --                           id, itype, from, to, neuronID, ntype
      table.insert(o.innovations, innovation)
    end

    for n=1,outputs do
      local innovation = Innovation(inputs+n, "new_neuron", -1, -1, inputs+n, "output")
      --                                  id, itype, from, to, neuronID, ntype
      table.insert(o.innovations, innovation)
    end

    return setmetatable(o, Population.mt)
  end
})

function Population:nextNeuronID()
  for n=#self.innovations, 1, -1 do
    local innovation = self.innovations[n]

    if innovation.itype == "new_neuron" then
      return innovation.neuronID+1
    end
  end
end

--------------------------------
--       PUBLIC FUNCTIONS     --
--------------------------------

function LuaNEAT.newPopulation(size, inputs, outputs)
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

  return Population(size, inputs, outputs)
end





-- public
size = 200
inputs = 3
outputs = 1

population = LuaNEAT.newPopulation(size, inputs, outputs)
--[[for _,i in ipairs(population.innovations) do
  print(tostring(i).."\n")
end]]

genome = Genome.basicGenome(1, 3, 1)
for _,neuron in ipairs(genome.NeuronGeneList) do
  print(tostring(neuron).."\n")
end



--print()





return LuaNEAT
