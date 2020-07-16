local demo = require"demos/cart-pole"

function love.load()
  demo.load()
end

function love.keypressed(key)
  if demo.keypressed then
    demo.keypressed(key)
  end
end

function love.update(dt)
  demo.update(dt)
end

function love.draw()
  demo.draw()
end
