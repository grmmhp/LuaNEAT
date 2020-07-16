LuaNEAT = require"LuaNEAT"

TheGrid = {}

local pool = LuaNEAT.newPool(50, 3, 3)

local creatures = {radius = 10, velocity = 4, angularVelocity = math.rad(5), maxLife = 300, sensorLength = 150, drawPointer = false, drawNearest = false}
local particles = {radius = 5, amount = 75, spawnAttempts = 100, timer = 0, spawnTime = 15}

local gridSize = 100
local drawMode = 0

local speed = 1

--

local function distance(creature, particle)
  return math.sqrt((creature.x-particle.x)^2+(creature.y-particle.y)^2)
end

local function newCreature(brain)
  local creature = {}

  creature.x = love.math.random(0, 799)
  creature.y = love.math.random(0, 599)
  creature.angle = love.math.random()*2*math.pi
  creature.life = creatures.maxLife
  creature.brain = brain

  return creature
end

local function moveCreature(creature, velocity)
  creature.x = creature.x + velocity*math.cos(creature.angle)
  creature.y = creature.y + velocity*math.sin(creature.angle)
end

local function rotateCreature(creature, angle)
  creature.angle = creature.angle + angle

  if creature.angle < 0 then
    creature.angle = 2*math.pi +   creature.angle
  end

  creature.angle = creature.angle%(2*math.pi)
end

local function checkCollision(creature, particle)
  if not particle then return; end

  if distance(creature, particle) < creatures.radius + particles.radius then
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
  local dy = creature.y-particle.y

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

      -- inputting the neural network
      if nearest then
        local inputs = {
          distance(creature, nearest)/creatures.sensorLength,
          getQuadrantAngle(creature)/math.pi,
          getAngle(creature, nearest)/math.pi,
          --[[creature.x/800,
          creature.y/800,
          nearest.x/800,
          nearest.y/600,]]
        }

        local outputs = creature.brain:forward(inputs)
        moveCreature(creature, outputs[1]*creatures.velocity)
        rotateCreature(creature, -outputs[2]*creatures.angularVelocity)
        rotateCreature(creature, outputs[3]*creatures.angularVelocity)

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

      if creature.life == 0 then
        table.remove(creatures, n)
      else
        creature.brain:incrementFitness(1)
      end
    end

    -- next generation
    if #creatures == 0 then
      next()
    end

    if #particles < particles.amount then
      particles.timer = particles.timer + 1

      if particles.timer > particles.spawnTime then
        spawnParticle()
        particles.timer = 0
      end
    end
  end
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
    pool:getBestPerformer():draw(200, 200, -19, 380)
  end
end

return TheGrid
