local neat = require"LuaNEAT"
local grapher = require "grapher"

local snake_demo = {
  game_window = 500,--love.graphics.getHeight(), -- size of the game window to draw
  grid_x = 0, -- x and y coordinates of the top left square of the grid
  grid_y = 0,
  grid_border = .1, -- percentage of the grid that will be its border (number between 0 and 1)

  grid_size = 30, --walls are at x=1, x=grid_size, y=1, y=grid_size

  waiting_time = 0, -- waiting time (in seconds) to change frame

  num_trials = 5, -- number of trials each network will play the game per generation

  mode = "bot", -- "user" or "bot";
}

local margin = 30
local width = snake_demo.game_window*1.3 - margin
local height = snake_demo.game_window - margin


-- initializing LuaNEAT
local population_size = 500--300
local num_inputs = 24
local num_outputs = 4
local neat_pool
local neural_nets_list = {}
local load = false
local save_when_generation_is = 20

if load then
  neat_pool = neat.load("snake_gen103_poolsize1000")
  population_size = neat_pool.size
else
  neat_pool = neat.newPool(population_size, num_inputs, num_outputs) -- the first generation genomes will be disconnected--neat.load("snake")
end

neat_pool.parameters.initialWeightRange = 30

neat_pool.parameters.mutation_rates.loopedLink = 0
neat_pool.parameters.mutation_rates.addLink = 5--0.988
neat_pool.parameters.mutation_rates.addNode = .15-- 0.085
neat_pool.parameters.mutation_rates.addBias = .2-- 0.085
neat_pool.parameters.mutation_rates.perturbWeight = 0.460
neat_pool.parameters.mutation_rates.replaceWeight = 0.0245
neat_pool.parameters.mutation_rates.weightStep = 0.825
neat_pool.parameters.mutation_rates.enableDisable = 0.0138

neat_pool.parameters.excessGenesCoefficient   = 1
neat_pool.parameters.disjointGenesCoefficient = 1
neat_pool.parameters.matchingGenesCoefficient = 0.4
neat_pool.parameters.sameSpeciesThreshold     = 3

--[[neat_pool:setInitialHiddenLayers(15, 15)
neat_pool.parameters.addLink = 0
neat_pool.parameters.addNode = 0]]

local stats = grapher.newGraph()
stats.x_label = "Generation"
stats.y_label = "Fitness"
stats.lower_x_range = 1
stats.upper_x_range = 2
stats.lower_y_range = 0
stats.upper_y_range = 40
stats.x_tick_marks_spacing = 1
stats.y_tick_marks_spacing = 5


stats:render(width-margin*3, height - 50)


-- some math functions
local function map(value, x0, x1, y1, y2)
  return (value-x0)/(x1-x0)*(y2-y1) + y1
end

local function clamp(value, x0, x1, y1, y2)
  value = math.min(value, x1)
  value = math.max(value, x0)

  return map(value, x0, x1, y1, y2)
end

local snake = {
  body = {}, --vector containing all body parts; first element is the head and last the tail; (x, y) pos
  dx = 1,
  dy = 0,

  energy = math.floor((snake_demo.grid_size-1)^2/2),
  fruits_eaten = 0,
  fitness_list = {},

  evaluating_brain = 1,
  trials = 1,
}

snake.resetEnergy = function()
  snake.energy = math.floor((snake_demo.grid_size-1)^2/2)
end

local fruit = {
  x,
  y,
}

-- control variables
local game_timer = 0

-- basic stuff
local function spawn_fruit()
  local superposition
  repeat -- randomly select a square to put the fruit on and repeats this process until the position doesnt coincides with the snake body
    superposition = false
    fruit.x = love.math.random(2, snake_demo.grid_size-1)
    fruit.y = love.math.random(2, snake_demo.grid_size-1)

    for i=1, #snake.body do
      if  fruit.x == snake.body[i][1]
      and fruit.y == snake.body[i][2] then
        superposition = true
      end
    end
  until superposition == false
end

local function initialize_game()
  game_timer = 0

  local x0 = 4-- snake's head initial position
  local y0 = math.floor((snake_demo.grid_size)/2)

  snake.body = {{x0, y0}, {x0-1, y0}}
  snake.dx = 1
  snake.dy = 0

  snake.fitness_list[snake.trials] = snake.fruits_eaten

  snake.trials = snake.trials + 1
  if snake.trials > snake_demo.num_trials then
    local fitness = 0

    for i=1, #snake.fitness_list do
      fitness = fitness + snake.fitness_list[i]
    end

    fitness = 5*fitness/#snake.fitness_list
    neural_nets_list[snake.evaluating_brain]:setFitness(fitness)

    snake.fitness_list = {}

    snake.trials = 1
    snake.evaluating_brain = snake.evaluating_brain + 1

    if snake.evaluating_brain > population_size then
      --print(neat_pool:nextGeneration())
      neat_pool:nextGeneration()
      neural_nets_list = neat_pool:getNeuralNetworks()

      if neat_pool.generation == save_when_generation_is then neat.save(neat_pool, "snake") end

      stats:newArrayPlot(neat_pool:getTopFitnessPoints(), "top_fitness", {1,1,0,1})
      stats.plots["top_fitness"].line_thickness = 3
      stats.lower_x_range = math.max(1, neat_pool:getGeneration()-1-50)
      stats.upper_x_range = neat_pool:getGeneration()-1
      stats.lower_y_range = 0
      stats.upper_y_range = math.max(40, neat_pool.statistics:getAllTimeTopFitness())

      stats.x_tick_marks_spacing = 10
      stats.y_tick_marks_spacing = 5

      stats:render(width-margin*3, height - 50)

      if neat_pool.generation == 2 then
        neat_pool:setParameterToEveryone("addLink", 0.988)
      end


      --print("Best fitness from generation ".. neat_pool:getGeneration()-1 .. " is ".. neat_pool:getLastBestFitness())

      snake.evaluating_brain = 1
    end
  end

  snake.resetEnergy()
  snake.fruits_eaten = 0

  spawn_fruit()
end

local function snake_look(dx, dy)
  local x = snake.body[1][1] + dx
  local y = snake.body[1][2] + dy

  local found_food = -1
  local found_body = -1
  local distance_to_wall

  --print("looking at direction (".. dx .. ", ".. dy.. ")")
  --print("snake is at (".. snake.body[1][1].. ", ".. snake.body[1][2].. ")")

  while true do
    for i=2,#snake.body do
      if (x == snake.body[i][1] and y == snake.body[i][2]) and (found_body == -1) then
        --print("body at (".. x.. ", ".. y.. ")")
        found_body = 1;
        break
      end
    end

    if x == fruit.x and y == fruit.y and (found_food == -1) then
      --print("food at (".. x.. ", ".. y.. ")")
      found_food = 1;
    end

    if x==1 or x==snake_demo.grid_size then
      distance_to_wall = 1/(math.abs(snake.body[1][1]-x) + math.abs(snake.body[1][2]-y))


      --print("wall at (".. x.. ", ".. y.. ")\n\n")
      return {found_body, found_food, distance_to_wall}
    end

    if y==1 or y==snake_demo.grid_size then
      distance_to_wall = 1/((math.abs(snake.body[1][1]-x) + math.abs(snake.body[1][2]-y))/(snake_demo.grid_size-1))


      --print("wall at (".. x.. ", ".. y.. ")\n\n")
      return {found_body, found_food, distance_to_wall}
    end

    x = x + dx
    y = y + dy
  end

  return -1 --not found
end

function snake_demo.load()
  love.window.setMode(snake_demo.game_window*3.5, snake_demo.game_window)

  -- initializing the neural nets pool
  if not load then neat_pool:initialize(true) end
  neural_nets_list = neat_pool:getNeuralNetworks()

  -- immediately initializes game
  initialize_game()
end

-- game logic
function snake_demo.update(dt)
  -- snake controls
  local direction-- = snake.direction

  if snake_demo.mode == "user" then
    if love.keyboard.isDown({"w", "up"}) then
      direction = "up"
    end
    if love.keyboard.isDown({"a", "left"}) then
      direction = "left"
    end
    if love.keyboard.isDown({"s", "down"}) then
      direction = "down"
    end
    if love.keyboard.isDown({"d", "right"}) then
      direction = "right"
    end
  end

  -------------------------------------------------------
  game_timer = game_timer + dt
  if game_timer < snake_demo.waiting_time then return end
  -------------------------------------------------------

  if snake_demo.mode == "bot" then
    local fx = fruit.x
    local fy = fruit.y
    local sx = snake.body[1][1]
    local sy = snake.body[1][2]

    --[[local angle_fruit = math.atan2(sy-fy, fx-sx) --this function returns an angle between -180 and 180 degrees
    local dist_fruit = math.abs(fx-sx)+math.abs(fy-sy)

    local signed_dist_x = sx-fx
    local signed_dist_y = sy-fy

    print(fx-sx,
          fy-sy,
          dist_fruit,
          snake_demo.grid_size)

    -- normalizing the inputs
    angle_fruit = map(angle_fruit, -math.pi, math.pi, -1, 1)
    dist_fruit = map(dist_fruit, 0, snake_demo.grid_size-2, -1, 1)

    signed_dist_x = map(signed_dist_x, -snake_demo.grid_size+2, snake_demo.grid_size-2, -1, 1)
    signed_dist_y = map(signed_dist_y, -snake_demo.grid_size+2, snake_demo.grid_size-2, -1, 1)]]

    -- outputs are to be interpreted as "forward, move left, move right" respectively. the selected action will be according to which neuron has the highest value
    --local outputs = neural_nets_list[snake.evaluating_brain]:forward(angle_fruit, signed_dist_x, signed_dist_y)


    inputs = {
      snake_look(0, -1)[1],
      snake_look(0, 1)[1],
      snake_look(-1, 0)[1],
      snake_look(1, 0)[1],
      snake_look(-1, -1)[1],
      snake_look(-1, 1)[1],
      snake_look(1, -1)[1],
      snake_look(1, 1)[1],

      snake_look(0, -1)[2],
      snake_look(0, 1)[2],
      snake_look(-1, 0)[2],
      snake_look(1, 0)[2],
      snake_look(-1, -1)[2],
      snake_look(-1, 1)[2],
      snake_look(1, -1)[2],
      snake_look(1, 1)[2],

      snake_look(0, -1)[3],
      snake_look(0, 1)[3],
      snake_look(-1, 0)[3],
      snake_look(1, 0)[3],
      snake_look(-1, -1)[3],
      snake_look(-1, 1)[3],
      snake_look(1, -1)[3],
      snake_look(1, 1)[3],
    }

    local outputs = neural_nets_list[snake.evaluating_brain]:forward(inputs)



    local maxval = math.max(outputs[1], outputs[2], outputs[3], outputs[4])
    local up = (outputs[1] == maxval)
    local down = (outputs[2] == maxval)
    local left = (outputs[3] == maxval)
    local right = (outputs[4] == maxval)

    if up then direction = "up" end
    if down then direction = "down" end
    if left then direction = "left" end
    if right then direction = "right" end
  end

  ----------------

  if direction == "up" then
    -- only moves up if already not moving down
    if snake.dy == 0 then
      snake.dx = 0
      snake.dy = -1
    end
  end
  if direction == "left" then
    -- only moves left if already not moving right
    if snake.dx == 0 then
      snake.dx = -1
      snake.dy = 0
    end
  end
  if direction == "down" then
    -- only moves down if already not moving up
    if snake.dy == 0 then
      snake.dx = 0
      snake.dy = 1
    end
  end
  if direction == "right" then
    -- only moves right if already not moving left
    if snake.dx == 0 then
      snake.dx = 1
      snake.dy = 0
    end
  end

  -- move the snake
  new_x = snake.body[1][1]+snake.dx
  new_y = snake.body[1][2]+snake.dy

  local game_over = false

  table.insert(snake.body, 1, {new_x, new_y})

  -- checking for collisions
  -- fruit collision
  if new_x == fruit.x and new_y == fruit.y then
    spawn_fruit()
    snake.fruits_eaten = snake.fruits_eaten + 1
    snake.resetEnergy()
  else
    table.remove(snake.body)
  end
  --table.remove(snake.body)

  -- collision with itself
  for i=2, #snake.body do
    if new_x == snake.body[i][1] and new_y == snake.body[i][2] then
      game_over = true
      break
    end
  end

  -- collision with walls
  if new_x == 1 or new_x == snake_demo.grid_size or new_y == 1 or new_y == snake_demo.grid_size then
    game_over = true
  end


  snake.energy = snake.energy - 1
  if snake.energy == 0 then game_over = true end

  if game_over then
    --neural_nets_list[snake.evaluating_brain]:setFitness(snake.fruits_eaten)
    initialize_game()
  end



  --[[for i, coord in ipairs(snake.body) do
    coord[1] = coord[1] + snake.dx
    coord[2] = coord[2] + snake.dy
  end]]

  -- resets timer
  game_timer = 0
end

function snake_demo.keypressed(key)
  if key=="r" then
    if snake_demo.waiting_time == 0 then
      snake_demo.waiting_time = .1
    elseif snake_demo.waiting_time == .1 then
      snake_demo.waiting_time = 0
    end
  elseif key=="s" then
    neat.save(neat_pool, "snake")
  end
end

function snake_demo.draw()
  local block_size = snake_demo.game_window/snake_demo.grid_size

  love.graphics.push()
  love.graphics.translate(snake_demo.grid_x, snake_demo.grid_y)

  local gap = block_size*snake_demo.grid_border
  local square_side = block_size-(2*gap)

  -------------------
  -- DRAWING WALLS --
  -------------------

  -- top and bottom wall
  love.graphics.setColor(1,1,1,1)
  for i=0, snake_demo.grid_size-1 do
    local x = i*block_size
    love.graphics.rectangle("fill", x+gap, gap, square_side, square_side)
    love.graphics.rectangle("fill", x+gap, block_size*(snake_demo.grid_size-1) + gap, square_side, square_side)
  end

  -- left and right walls
  for i=0, snake_demo.grid_size-1 do
    local y = i*block_size
    love.graphics.rectangle("fill", gap, y+gap, square_side, square_side)
    love.graphics.rectangle("fill", block_size*(snake_demo.grid_size-1) + gap, y+gap, square_side, square_side)
  end


  -----------------------
  -- DRAWING THE FRUIT --
  -----------------------

  love.graphics.setColor(1,0,0,1)
  love.graphics.rectangle("fill", (fruit.x-1)*block_size+gap, (fruit.y-1)*block_size+gap, square_side, square_side)


  -----------------------
  -- DRAWING THE SNAKE --
  -----------------------

  for i=1, #snake.body do
    local x = (snake.body[i][1]-1)*block_size
    local y = (snake.body[i][2]-1)*block_size

    if i==1 then love.graphics.setColor(0,1,0,1) else love.graphics.setColor(0,.5,0,1) end
    love.graphics.rectangle("fill", x+gap, y+gap, square_side, square_side)
  end

  love.graphics.pop()

  love.graphics.print(
    "Generation ".. neat_pool:getGeneration() ..
    "\n".. neat_pool:getSpeciesAmount() .. " species" ..
    "\n\nCandidate #".. snake.evaluating_brain..
    "\nTrial #".. snake.trials ..
    "\nFitness: ".. 5*snake.fruits_eaten ..
    "\nEnergy: ".. snake.energy,
    50,
    50
  )



  -- draw the neural net
  local net = neural_nets_list[snake.evaluating_brain]

  net:draw(snake_demo.game_window + margin, margin, width, height - 50)
  stats:draw(snake_demo.game_window + margin + width + margin, margin)
end


return snake_demo
