require "LuaNEAT"
local demo = require"demos/snake/snake"

function love.load()
  if demo.load then demo.load() end

  running = false
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
  end
end

function love.update(dt)
  if (not running) and (not step) then return end

  demo.update(dt)

  step = false
end

function love.draw()
  demo.draw()
end
