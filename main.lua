NEAT = require"LuaNEAT"

print(NEAT._VERSION)

function love.load()
end

function love.update(dt)
end
--run()
function love.draw()
  --genome:drawNeuralNetwork(200, 200, 30, 30)
  --genome2:drawNeuralNetwork(200, 200, 200+30+10, 30)
  --offspring:drawNeuralNetwork(200, 200, 200+200+30+10, 30)
  pool.species[1].leader:drawNeuralNetwork(200, 200, 30, 30)
  pool.species[2].leader:drawNeuralNetwork(200, 200, 200+30+10, 30)
  pool.species[3].leader:drawNeuralNetwork(200, 200, 200+200+30+10, 30)
end
