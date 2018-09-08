NEAT = require"NEAT"



function love.load()
end

function love.update(dt)
end
--run()
function love.draw()
  genome:drawNeuralNetwork(200, 200, 30, 30)
  genome2:drawNeuralNetwork(200, 200, 200+30+10, 30)
  offspring:drawNeuralNetwork(200, 200, 200+200+30+10, 30)
end
