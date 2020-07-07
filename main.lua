local demo = require"demos/cart-pole"

function love.load()
  demo.load()
end

function love.update(dt)
  demo.update(dt)
end

function love.draw()
  demo.draw()
end
