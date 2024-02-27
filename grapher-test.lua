grapher = require "grapher"
local gt = {}

local function graph_f(f, x0, x1, n)
  local t = {}

  for i=1,n+1 do
    local step = (x1-x0)/n
    local x = x0 + step*(i-1)
    t[2*i-1] = x
    t[2*i] = f(x)

    print(x, f(x))
  end

  return t
end

local function square(x) return x^2 end

local parabola = graph_f(math.exp, -2, 2, 10)





local graph = grapher.newGraph(-2, 2, -2, 2)
graph:newPlot(parabola, "parabola", {1, 0, 0, 1})

graph.x_label = "Generation"
graph.y_label = "Fitness"

function gt.draw()
  graph:render(200, 200, 300, 300)
end

return gt
