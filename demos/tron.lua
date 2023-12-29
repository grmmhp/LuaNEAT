LuaNEAT = require"LuaNEAT"

TheGrid = {}

local pool = LuaNEAT.newPool(100, 1, 2, false)

local creatures = {radius = 10, maxVelocity = 100, maxAngularVelocity=1, angularVelocity = math.rad(5), maxLife = 300, sensorLength = 150, drawPointer = false, drawNearest = false}
local particles = {radius = 5, amount = 100, spawnAttempts = 100, timer = 0, spawnTime = 15}

local gridSize = 100
local drawMode = 0

local speed = 1

-- some math functions
local function map(value, x0, x1, y1, y2)
  return (value-x0)/(x1-x0)*(y2-y1) + y1
end

local function clamp(value, x0, x1, y1, y2)
  value = math.min(value, x1)
  value = math.max(value, x0)

  return map(value, x0, x1, y1, y2)
end

local function distance(creature, particle)
  return math.sqrt((creature.x-particle.x)^2+(creature.y-particle.y)^2)
end

local function newCreature(brain)
  local creature = {}

  creature.x = love.math.random(0, 799)
  creature.y = love.math.random(0, 599)
  creature.angle = love.math.random()*2*math.pi
  creature.angularVelocity = 0
  creature.angularAcceleration = 0

  creature.velocityX = 0
  creature.velocityY = 0

  creature.accelerationX = 0
  creature.accelerationY = 0


  creature.velocity=creatures.maxVelocity
  creature.acceleration=0
  creature.life = creatures.maxLife
  creature.brain = brain

  return creature
end

local function moveCreature(creature, dt)
  --creature.velocity = creature.velocity + math.sqrt(creature.accelerationX^2 + creature.accelerationY^2)*dt--creature.acceleration*dt
  creature.velocityX = creature.velocityX + creature.accelerationX*dt
  creature.velocityY = creature.velocityY + creature.accelerationY*dt

  local vel = math.sqrt(creature.velocityX^2 + creature.velocityY^2)
  if vel > creatures.maxVelocity then creature.velocityX = creature.velocityX/creatures.maxVelocity; creature.velocityY = creature.velocityX/creatures.maxVelocity end

  creature.x = creature.x + creature.velocityX*dt--*math.cos(creature.angle)*dt
  creature.y = creature.y + creature.velocityY*dt--math.sin(creature.angle)*dt
end

local function rotateCreature(creature, dt)--, angle)
  creature.angularVelocity = creature.angularVelocity + creature.angularAcceleration*dt

  if creature.angularVelocity > creatures.maxAngularVelocity then creature.angularVelocity = creatures.maxAngularVelocity end

  creature.angle = creature.angle + creature.angularVelocity*dt--angle

  if creature.angle < 0 then
    creature.angle = 2*math.pi +   creature.angle
  end

  creature.angle = creature.angle%(2*math.pi)
end

local function checkCollision(creature, particle)
  if not particle then return; end

  if distance(creature, particle) < creatures.radius + particles.radius then
    creature.brain:incrementFitness(10)
    return true
  end

  return false
end

local function getNearestParticle(creature)
  local nearest
  local index

  for i, particle in ipairs(particles) do
    if not nearest then
      nearest = particle
      index = i
    else
      if distance(creature, particle) < distance(creature, nearest) then
        nearest = particle
        index = i
      end
    end
  end

  return nearest, index
end

local function getAngle(creature, particle)
  local dx = creature.x-particle.x
  local dy = particle.y-creature.y

  return math.atan2(dy, dx)
end

local function getQuadrantAngle(creature)
  local theta = creature.angle

  return math.atan2(math.sin(theta), math.cos(theta))
end

local function drawCreature(creature)
  local alive_color = {254/255, 218/255, 9/255}
  local dead_color = {228/255, 74/255, 12/255}
  local glow_color = {}

  local amount = creature.life/creatures.maxLife
  for n = 1, 3 do
    glow_color[n] = amount*alive_color[n] + (1-amount)*dead_color[n]
  end

  local glow_amount = 10
  local glow_level = .35 --a number from 0 to 1

  local initial_radius = 1.1 -- the initial radius
  local max_radius = 2

  for n = 1, glow_amount do
    local mult = max_radius + (n-1)*(initial_radius-max_radius)/(glow_amount-1)

    love.graphics.setColor(glow_color[1], glow_color[2], glow_color[3], glow_level/glow_amount)
    love.graphics.circle("fill", creature.x, creature.y, creatures.radius*mult)
  end

  -- draw the little vector
  if creatures.drawPointer then
    local length = 50
    love.graphics.setColor(glow_color[1], glow_color[2], glow_color[3], 1)
    love.graphics.line(creature.x, creature.y, creature.x+length*math.cos(creature.angle), creature.y+length*math.sin(creature.angle))
  end

  --

  love.graphics.setColor(0, 0, 0, 1)
  --love.graphics.circle("fill", creature.x, creature.y, creatures.radius)

  love.graphics.setColor(glow_color[1], glow_color[2], glow_color[3], 1)
  love.graphics.circle("line", creature.x, creature.y, creatures.radius)
end

local function newParticle()
  local particle = {}

  particle.x = love.math.random(0, 799)
  particle.y = love.math.random(0, 599)

  return particle
end

local function spawnParticle()
  for attempts = 1, particles.spawnAttempts do
    local particle = newParticle()
    local flag

    for _, creature in ipairs(creatures) do
      if checkCollision(creature, particle) then
        flag = true

        break
      end
    end

    if not flag then
      table.insert(particles, particle)

      return
    end
  end
end

local function spawnAllParticles()
  for n = #particles, particles.amount do
    spawnParticle()
  end
end

local function drawParticle(particle)
  local glow_color = {116/255, 249/255, 246/255}
  local glow_amount = 10
  local glow_level = .25 --a number from 0 to 1

  local initial_radius = 1.1 -- the initial radius
  local max_radius = 5

  for n = 1, glow_amount do
    local mult = max_radius + (n-1)*(initial_radius-max_radius)/(glow_amount-1)

    love.graphics.setColor(glow_color[1], glow_color[2], glow_color[3], glow_level/glow_amount)
    love.graphics.circle("fill", particle.x, particle.y, particles.radius*mult)
  end

  --

  love.graphics.setColor(0, 0, 0, 1)
  --love.graphics.circle("fill", particle.x, particle.y, particles.radius)

  love.graphics.setColor(116/255, 249/255, 246/255, 1)
  love.graphics.circle("line", particle.x, particle.y, particles.radius)
end

local function next()
  pool:nextGeneration()

  while #particles > 0 do
    table.remove(particles)
  end

  local nets = pool:getNeuralNetworks()
  for _, net in ipairs(nets) do
    table.insert(creatures, newCreature(net))
  end

  spawnAllParticles()
end

--

function TheGrid.load()
  grid = love.graphics.newImage("img/grid.png")

  pool:initialize()

  local nets = pool:getNeuralNetworks()
  for _, net in ipairs(nets) do
    table.insert(creatures, newCreature(net))
  end

  spawnAllParticles()
end

function TheGrid.keypressed(key)
  if key == "d" then
    drawMode = (drawMode+1)%4
  elseif key == "s" then
    speed = speed * 2

    if speed > 2048 then
      speed = 1
    end
  end
end

function TheGrid.update(dt)
  for s = 1, speed do
    for n = #creatures, 1, -1 do
      local creature = creatures[n]
      local nearest, index = getNearestParticle(creature)

      local dx, dy

      if not nearest then
        dx=0
        dy=0
      else
        if creature.x > nearest.x then dx=1 else dx=-1 end
        if creature.y > nearest.y then dy=1 else dy=-1 end
      end

      -- inputting the neural network
      if nearest then
        local inputs = {
          distance(creature, nearest)/creatures.sensorLength,
          --getQuadrantAngle(creature)/math.pi,
          map(getAngle(creature, nearest), -math.pi, math.pi, -1, 1),
          --dx,
          --dy,
          --map(creature.velocity, 0, creatures.maxVelocity, -1, 1),
          --[[creature.x/800,
          creature.y/800,
          nearest.x/800,
          nearest.y/600,]]
        }

        local outputs = creature.brain:forward(inputs)
        --creature.x = creature.x + (2*outputs[1]-1)*creatures.velocity*50*dt
        --creature.y = creature.y + (2*outputs[2]-1)*creatures.velocity*50*dt
        --creature.angularAcceleration = (2*outputs[1]-1)*100
        --creature.acceleration = (2*outputs[2]-1)*50
        local acc = 20--outputs[3]*20
        creature.accelerationX = acc*outputs[1]
        creature.accelerationY = acc*outputs[2]
        moveCreature(creature, dt)--outputs[1]*creatures.velocity)
        --rotateCreature(creature, dt)--(2*outputs[2]-1)*creatures.angularVelocity)
        --creature.angle = outputs[1]*2*math.pi
        --rotateCreature(creature, outputs[3]*creatures.angularVelocity)

        --[[if outputs[1] > .5 then
          moveCreature(creature, creatures.velocity)
        end

        if outputs[2] > .5 then
          rotateCreature(creature, creatures.angularVelocity)
        else
          rotateCreature(creature, creatures.angularVelocity)
        end]]

        --[[local dx = outputs[1]
        local dy = outputs[2]
        local mag = math.sqrt(dx^2+dy^2)

        if outputs[3] > .5 then
          creature.x = creature.x + creatures.velocity*dx/mag
        else
          creature.x = creature.x - creatures.velocity*dx/mag
        end

        if outputs[4] > .5 then
          creature.y = creature.y - creatures.velocity*dy/mag
        else
          creature.y = creature.y + creatures.velocity*dy/mag
        end]]

        --moveCreature(creature, outputs[1]*creatures.velocity)
        --rotateCreature(creature, outputs[2]*creatures.angularVelocity*2 - creatures.angularVelocity)
      end

      creature.life = creature.life - 1

      if checkCollision(creature, nearest) then
        creature.life = creatures.maxLife

        --creature.brain:incrementFitness(1)
        table.remove(particles, index)
      end

      maxfit=0

      if creature.life == 0 then
        table.remove(creatures, n)
      else
        --creature.brain:incrementFitness(10)
      end

      if creature.brain:getFitness() > maxfit then
        maxfit = creature.brain:getFitness()
      end

    end

    -- next generation
    if #creatures == 0 then
      next()
      --maxfit=0

      --[[print("creatures resetted\nfitnesses:\n")
      for n=1,#creatures do
        print(creatures[n].brain:getFitness())
      end
      error()]]
    end

    if #particles < particles.amount then
      particles.timer = particles.timer + 1

      if particles.timer > particles.spawnTime then
        spawnParticle()
        particles.timer = 0
      end
    end
  end

  --print("max fitness is ".. maxfit)
end

function TheGrid.draw()
  -- drawing the background
  love.graphics.setColor(42/255, 107/255, 110/255, .5)

  for y = 0, math.ceil(600/(gridSize*2)) do
    for x = 0, math.ceil(800/(gridSize*2)) do
      love.graphics.draw(grid, x*(gridSize*2) - 3*gridSize/2, y*(gridSize*2) - 3*gridSize/2, 0, (gridSize*2)/grid:getWidth(), (gridSize*2)/grid:getHeight())
    end
  end

  love.graphics.setColor(42/255, 107/255, 110/255, 1)

  for y = 0, math.ceil(600/gridSize) do
    for x = 0, math.ceil(800/gridSize) do
      love.graphics.draw(grid, x*gridSize, y*gridSize, 0, gridSize/grid:getWidth(), gridSize/grid:getHeight())
    end
  end

  -- drawing the particles
  for _, particle in ipairs(particles) do
    drawParticle(particle)
  end

  -- drawing the creatures
  for _, creature in ipairs(creatures) do
    if creatures.drawNearest then
      local nearest = getNearestParticle(creature)
      love.graphics.setColor(0, 1, 0, 1)
      love.graphics.line(creature.x, creature.y, nearest.x, nearest.y)
    end

    drawCreature(creature)
  end

  -- drawing info
  if drawMode == 1 or drawMode == 3 then
    love.graphics.setColor(0, 0, 0, .25)
    love.graphics.rectangle("fill", 5, 5, 95, 80)

    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.print("Generation ".. pool.generation, 10, 10)
    if #creatures == 1 then
      love.graphics.print(#creatures .. " creature", 10, 35)
    else
      love.graphics.print(#creatures .. " creatures", 10, 35)
    end
    love.graphics.print(#particles .. " particles", 10, 50)
    love.graphics.print(pool:getSpeciesAmount() .. " species", 10, 65)
    love.graphics.print(speed .. "x speed", 10, 80)
    love.graphics.print(love.timer.getFPS() .. " FPS", 10, 95)
  end

  if drawMode == 2 or drawMode == 3 then
    -- drawing the best net
    love.graphics.setColor(0, 0, 0, .25)
    love.graphics.rectangle("fill", 5, 350+19-5, 200-53+5, 232)
    pool:getBestPerformer():draw(200, 200, 350-19, 380)
  end
end

return TheGrid
