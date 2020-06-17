LuaNEAT = require"LuaNEAT"

print(LuaNEAT.version())

function newBird(brain)
  local bird = {}

  bird.x = 400
  bird.y = 300
  bird.jump = true
  bird.brain = brain
  bird.timer = 0
  bird.velocity = 0

  return bird
end

function drawBird(bird)
  love.graphics.setColor(249/255, 241/255, 36/255, .25)
  love.graphics.circle("fill", bird.x, bird.y, birds.radius)

  love.graphics.setColor(0, 0, 0, .25)
  love.graphics.circle("line", bird.x, bird.y, birds.radius)
end

function newPipePair(x)
  local pipe = {}

  pipe.x = x
  pipe.y = love.math.random(50, 600-pipes.gap-50)

  return pipe
end

function drawPipePair(pipePair, red)
  love.graphics.setColor(116/255, 191/255, 47/255, 1)
  if red then love.graphics.setColor(1, 0, 0, 1) end
  love.graphics.rectangle("fill", pipePair.x, 0, pipes.width, pipePair.y)
  love.graphics.rectangle("fill", pipePair.x, pipePair.y+pipes.gap, pipes.width, 600-pipePair.y)

  love.graphics.setColor(0, 0, 0, 1)
  love.graphics.rectangle("line", pipePair.x, -1, pipes.width, pipePair.y+1)
  love.graphics.rectangle("line", pipePair.x, pipePair.y+pipes.gap, pipes.width, 600-pipePair.y+1)
end

function checkCollision(bird, pipePair)
  if  bird.x+birds.radius > pipePair.x and bird.x-birds.radius < pipePair.x+pipes.width
  and (bird.y-birds.radius < pipePair.y or bird.y+birds.radius > pipePair.y+pipes.gap) then
    return true
  end

  return false
end

function getNearestPipe(bird)
  if not bird then return; end

  local nearest
  local index
  local closest_dist = math.huge

  for n = 1, #pipes do
    local pair = pipes[n]
    local dist = pair.x+pipes.width - bird.x

    if pair.x+pipes.width - bird.x < closest_dist and dist > 0 then
      nearest = pair
      closest_dist = pair.x+pipes.width - bird.x
      index = n
    end
  end

  return nearest, index
end

function reset()
  while #pipes>0 do
    table.remove(pipes)
  end

  pipes[1] = newPipePair(1000)

  local nets = pool:getNeuralNetworks()
  for n = 1, #nets do
    birds[n] = newBird(nets[n])
  end
end

--

function love.load()
  pool = LuaNEAT.newPool(300, 4, 1)
  pool:initialize()

  birds = {mass = 10, radius = 25, jumpForce = 15, jumpTime = 3}
  pipes = {width = 100, distance = 400, gap = 200, scrollVelocity = 7}

  gravity = 2.5

  pipes[1] = newPipePair(1000)

  local nets = pool:getNeuralNetworks()
  for n = 1, #nets do
    birds[n] = newBird(nets[n])
  end

  repeats = 1
end

function love.keypressed(key)
  if key == "r" then
    pool:nextGeneration()

    reset()
  end
end

function love.update(dt)
  for _ = 1, repeats do
    for n = #birds, 1, -1 do
      local bird = birds[n]

      bird.velocity = bird.velocity + gravity

      local nearest = getNearestPipe(bird)
      local inputs = {
        bird.y/600,
        nearest.y/600,
        (nearest.y+pipes.gap)/600,
        nearest.x/800,
      }

      local outputs = bird.brain:forward(inputs)

      if outputs[1] > .5 and bird.jump then--outputs[2] and bird.jump then
        bird.velocity = -birds.jumpForce

        bird.jump = false
      end

      bird.y = bird.y + bird.velocity
      bird.timer = bird.timer + 1

      if bird.timer > birds.jumpTime then
        bird.timer = 0
        bird.jump = true
      end

      if bird.y < 0 then
        bird.y = 0
        bird.velocity = 0
      elseif bird.y > love.graphics.getHeight()-1 then
        bird.y = love.graphics.getHeight()-1
        bird.velocity = 0

        if bird.brain:getFitness() > 0 then
          bird.brain:incrementFitness(-1)
        end

        table.remove(birds, n)
      end

      if checkCollision(bird, nearest) then
        table.remove(birds, n)
      else
        bird.brain:incrementFitness(1)
      end
    end

    for n = #pipes, 1, -1 do
      local pair = pipes[n]

      pair.x = pair.x - pipes.scrollVelocity

      if n == #pipes then
        if 800-pair.x > pipes.distance then
          table.insert(pipes, newPipePair(pair.x+pipes.distance))
        end
      end

      if pair.x+pipes.width < 0 then
        table.remove(pipes, n)
      end
    end

    if #birds == 0 then
      pool:nextGeneration()

      reset()
    end
    if pool.generation == 200 then
      repeats = 1
      break
    end
  end
end

function love.draw()
  love.graphics.setBackgroundColor(78/255, 192/255, 202/255)

  for n = 1, #birds do
    local bird = birds[n]
    drawBird(bird)
  end

  local _,index = getNearestPipe(birds[1])

  for n = 1, #pipes do
    local pipePair = pipes[n]
    local red = index == n
    drawPipePair(pipePair, red)
  end

  love.graphics.setColor(0, 0, 0, 1)
  love.graphics.print("Generation ".. pool.generation, 10, 10)

  if #birds == 1 then
    love.graphics.print(#birds .. " bird", 10, 25)
  else
    love.graphics.print(#birds .. " birds", 10, 25)
  end
  love.graphics.print(#pool.species .. " species", 10, 40)
  love.graphics.print("Best fitness is ".. pool:getBestFitness(), 10, 55)
  love.graphics.print(love.timer.getFPS() .. " FPS", 10, 70)


  local best = pool:getBestPerformer()
  best:draw(200, 200, 0, 375)
end
