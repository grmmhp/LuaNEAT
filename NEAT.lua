local LNEAT = {}

--------------------------------
  --       POPULATION         --
--------------------------------

local Population = {}
Population.mt = {__index = Population}

setmetatable(Population, {
  __call = function(t)
    local o = {}

    return setmetatable(o, Population.mt)
  end
})

--------------------------------
--           GENOME           --
--------------------------------

local Genome = {}
Genome.mt = {__index = Genome}

setmetatable(Genome, {
  __call = function(t, NeuronGeneList, LinkGeneList)
    local o = {}

    o.NeuronGeneList = NeuronGeneList
    o.LinkGeneList = LinkGeneList

    return setmetatable(o, Genome.mt)
  end
})

--------------------------------
--          GENES             --
--------------------------------

-- neuron gene
local NeuronGene = {}
NeuronGene.mt = {__index = NeuronGene}

setmetatable(NeuronGene, {
  __call = function(t, id)
    local o = {}
    o.id = id

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
    return  "Innovation: " .. self.innovation
          .."\nWeight: " .. self.weight
          .."\nFrom: " .. self.from
          .."\nTo: " .. self.to
          .."\nEnabled: " .. tostring(self.enabled)
          .."\nRecurrent: " .. tostring(self.recurrent)
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
  --        MUTATIONS         --
--------------------------------

local function mutateAddNode()
end

local function mutateAddLink()
end





-- public
linkgene1 = LinkGene(1, .75, 2, 6, true, false)
--print(tostring(linkgene1))
linkgene2 = linkgene1:copy()

linkgene1.innovation = 2
print(tostring(linkgene1) .. "\n\n" .. tostring(linkgene2))





return LNEAT
