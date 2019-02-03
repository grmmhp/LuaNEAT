NEAT = require"LuaNEAT"

print(NEAT._VERSION)

function love.load()
  stop = false
end

function love.update(dt)
  for _, species in ipairs(pool.species) do
    for __, genome in ipairs(species.genomes) do
      local sum=0
      for ___, link in ipairs(genome.LinkGeneList) do
        sum = sum + math.abs(link.weight)
      end
      local diff = math.abs(sum-100)
      genome.fitness = (1000/diff)
      --genome.fitness = 1/#genome.LinkGeneList
    end
  end

  if not stop then
    pool:nextGeneration()
  end
end

function love.draw()
  love.graphics.setBackgroundColor(.1, .1, .1, 1)
  love.graphics.print("Generation ".. pool.generation, 10, 10)
  love.graphics.print(#pool.species .. " species", 10, 30)
  love.graphics.setColor(1, 1, 0)
  love.graphics.print(love.timer.getFPS() .. " FPS", 10, 90)
  love.graphics.setColor(1, 1, 1)

  local best = pool.lastBestGenome or pool.species[1].genomes[1]
  local fit = best.fitness
  local sum = 0
  for _, link in ipairs(best.LinkGeneList) do
    sum = sum + math.abs(link.weight)
  end

  if math.abs(100-sum) < 0.001 then
    stop = true
  end

  love.graphics.print("top fit: ~".. math.floor(pool.topFitness*100)/100, 10, 50)
  love.graphics.print("weight sum: ~".. math.floor(sum*100)/100, 10, 70)
  best:drawNeuralNetwork(200, 200, 300, 200)
end
