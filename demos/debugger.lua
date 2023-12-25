local debugger = {}
local rand = 0

function debugger.load()
end

function debugger.update(dt)
  _NET:forward(-1, 0, 1)
  rand = love.math.random(0, 1)
end

function debugger.draw()
  _GEN:draw(50, 50, 700, 400)
  love.graphics.print(rand, 10, 10)
end

return debugger
