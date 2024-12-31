require "grapher"
local demo = require"demos/snake/snake"

function love.load()
  if demo.load then demo.load() end

  times_to_run = 1
  times_to_run_limit = 1024
  running = true
  step = false
end

function love.keypressed(key)
  if demo.keypressed then
    demo.keypressed(key)
  end

  if key == "space" then
    step = not step
  elseif key == "f1" then
    running = not running
  elseif key == "v" then
    times_to_run = times_to_run*2
    if times_to_run > times_to_run_limit then times_to_run = 1 end
  end
end

function love.update(dt)
  if (not running) and (not step) then return end

  for n=1, times_to_run do
    demo.update(dt)
  end

  step = false
end

function love.draw()
  demo.draw()
end
