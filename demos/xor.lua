local LuaNEAT = require"LuaNEAT"

local xor = {}

-- initializing LuaNEAT
local pool_size = 300
local inputs = 2 -- number of inputs
local outputs = 1 -- number of outputs
local pool = LuaNEAT.newPool(pool_size, inputs, outputs)

pool.parameters.loopedLink = 0
pool.parameters.addLink = 1
pool.parameters.addNode = 1
pool.parameters.enableDisable = 0
pool.parameters.excessGenesCoefficient   = 2
pool.parameters.disjointGenesCoefficient = 1
pool.parameters.matchingGenesCoefficient = 0.5

-- how many seconds until the next generation
local wait = 0--.25
local counter = 0

-- stop the simulation when the best net error is below %accuracy%
local stop=false
local accuracy=1e-4

-- the neural networks we will get from the pool
local neural_nets = {}


-- for drawing the stats
local margin = 50
local stats_height = 200


-- for each table in the training_set,
-- [1] and [2] are the inputs and
-- [3] the output ([1] xor [2])

local training_set = {
  --{0, 0, 0},
  --{0, 1, 1},
  --{1, 0, 1},
  --{1, 1, 0}
  {-1, -1, -1},
  {-1, 1, 1},
  {1, -1, 1},
  {1, 1, -1},
}

local function map(value, x0, x1, y1, y2)
  return (value-x0)/(x1-x0)*(y2-y1) + y1
end

function xor.load()
  -- initializing the pool
  pool:initialize()

  -- getting the first nets from the pool
  neural_nets = pool:getNeuralNetworks()
end

function xor.update(dt)
  if stop then return end

  if counter == 0 then
    -- evaluating the neural nets
    for n=1, #neural_nets do
      local sum=0
      for t=1, #training_set do
        local input1, input2 = training_set[t][1], training_set[t][2]

        local outputs = neural_nets[n]:forward("snapshot", input1, input2)

        local err = (outputs[1] - training_set[t][3])^2
        sum = sum + err
      end
      print(sum)
      neural_nets[n]:setFitness(8 - math.sqrt(sum))
    end
  end

  if counter > wait then
    counter = 0

    -- next generation
    pool:nextGeneration()
    -- getting the new neural nets
    neural_nets = pool:getNeuralNetworks()
  else
    counter = counter + dt
  end
end

function xor.draw()
  local w, h = love.graphics.getDimensions()
  -- drawing top fit graph
  local top_info = pool.statistics:getTopFitnessPoints()
  local top_points = {}
  local avg_info = pool.statistics:getAverageFitnessPoints()
  local avg_points = {}

  for n=1,#top_info do
    table.insert(top_points, map(n, 1, #top_info, margin, w-margin))
    table.insert(top_points, map(top_info[n], 0, 1, margin+stats_height, margin))
  end
  if #top_info > 1 then
    love.graphics.setColor(0,1,0,1)
    love.graphics.line(top_points)
  end

  -- drawing avg fit graph


  for n=1,#avg_info do
    table.insert(avg_points, map(n, 1, #avg_info, margin, w-margin))
    table.insert(avg_points, map(avg_info[n], 0, 1, margin+stats_height, margin))
  end
  if #avg_info > 1 then
    love.graphics.setColor(0,0,1,1)
    --love.graphics.line(avg_points)
  end

  -- graphs rectangles
  love.graphics.setColor(0,1,0,1)
  love.graphics.rectangle("line", margin, margin, w-margin*2, stats_height)
  love.graphics.setColor(1,1,1,1)
  love.graphics.print("Stats from the last ".. pool:getGeneration() .. " generations", margin, 30)

  -- bottom left and right rectangles
  love.graphics.setColor(0,1,0,1)
  love.graphics.rectangle("line", margin, margin*2+stats_height, w/2-margin-margin/2, h-margin*3-stats_height)
  love.graphics.rectangle("line", margin*2+w/2-margin-margin/2, margin*2+stats_height, w/2-margin-margin/2, h-margin*3-stats_height)

  -- drawing the best neural net
  local net = pool:getBestPerformer()
  local width = w/2-margin-margin/2
  local height = h-margin*3-stats_height
  net:draw(margin, margin*2+stats_height + 25, width, height - 50) -- x, y, width, height


  -- printing some stats
  local info_x = margin*2+w/2-margin-margin/2 + 10
  local info_y = margin*2+stats_height + 10

  -- getting the outputs from the best net
  local results = {}
  local total_error=0
  for t=1,#training_set do
    results[t] = net:forward("snapshot", training_set[t][1], training_set[t][2])[1]
    local err = (results[t] - training_set[t][3])^2
    total_error = total_error + err
  end
  total_error = math.sqrt(total_error)
  if total_error < accuracy then
    if not stop then
      LuaNEAT.save(pool, "xor")
    end
    stop=true
  end

  local worst_fit = math.huge
  local count=0
  for s=1,#pool.species do
    for g=1,#pool.species[s].genomes do
      if pool.species[s].genomes[g].fitness < worst_fit then
        worst_fit = pool.species[s].genomes[g].fitness
      end
      count=count+1
    end
  end

  love.graphics.setColor(1,1,1,1)
  love.graphics.print(
    "Generation ".. pool:getGeneration() .."\n" ..
    pool:getSpeciesAmount() .. " species" .. "\n" ..
    "Top fitness: ".. net:getFitness() .. "\n\n" ..
    "Worst fitness: ".. worst_fit.. "\n#Genomes:".. count .. "\n\n" ..
    "Best network:" .. "\n" ..
    training_set[1][1] .. " xor ".. training_set[1][2] .. " = ".. string.format("%.2f", results[1]) .. "\n" ..
    training_set[2][1] .. " xor ".. training_set[2][2] .. " = ".. string.format("%.2f", results[2]) .. "\n" ..
    training_set[3][1] .. " xor ".. training_set[3][2] .. " = ".. string.format("%.2f", results[3]) .. "\n" ..
    training_set[4][1] .. " xor ".. training_set[4][2] .. " = ".. string.format("%.2f", results[4]) .. "\n" ..
    "Error is ".. total_error,
    info_x, info_y
  )
end

return xor
