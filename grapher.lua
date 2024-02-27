local grapher = {}

local function affine_map(value, x0, x1, y1, y2)
  return (value-x0)/(x1-x0)*(y2-y1) + y1
end

local graph = {}
graph.mt = {__index = graph}

function grapher.newGraph(x0, x1, y0, y1)
  local obj = {

    border = true,
    border_thickness = 1,

    lower_x_range = x0 or 0,
    upper_x_range = x1 or 1,
    lower_y_range = y0 or 0,
    upper_y_range = y1 or 1,

    label_font = love.graphics.getFont(),
    x_label = "",
    y_label = "",

    show_tick_marks = true,
    tick_mark_size = 10,
    tick_mark_thickness=1,
    x_tick_marks_spacing=1,
    y_tick_marks_spacing=1,
    tick_mark_number_decimal_places=1,

    margin = 10,

    draw_x_axis = true,
    draw_y_axis = true,

    plots = {},
  }

  return setmetatable(obj, graph.mt)
end

function graph:newPlot(points, name, color)
  local plot = {
    points = points,

    line_color = color or {1,1,1,1}, --rgba
    line_thickness = 1,
  }

  self.plots[name] = plot
  return self.plots[name]
end

function graph:deletePlot(name)
  self.plots[name] = nil
end

function graph:newArrayPlot(array, name, color)
  local points = {}

  for x,y in ipairs(array) do
    table.insert(points, x)
    table.insert(points, y)
  end

  return self:newPlot(points, name, color)
end

function graph:newFunctionPlot(f, x0, x1, step, name, color)
  local t = {}

  local i=0
  local x = x0
  while x<x1 do
    x = x + step*i
    t[2*i-1] = x
    t[2*i] = f(x)

    i=i+1
  end

  return self:newPlot(t, name, color)
end

function graph:render(x, y, width, height)
  local canvas = love.graphics.newCanvas(width, height)
  love.graphics.setBlendMode("alpha", "premultiplied")
  love.graphics.setColor(1, 1, 1, 1)
  love.graphics.setCanvas(canvas)
  love.graphics.setBlendMode("alpha")

  -- axis
  if self.draw_x_axis then
    local y0 = affine_map(0, self.lower_y_range, self.upper_y_range, 0, height)
    love.graphics.line(0, y0, width, y0)
  end
  if self.draw_y_axis then
    local x0 = affine_map(0, self.lower_x_range, self.upper_x_range, 0, width)
    love.graphics.line(x0, 0, x0, height)
  end

  if self.draw_y_axis then
  end

  love.graphics.setColor(1,1,1,1)
  love.graphics.rectangle("line", 0, 0, width, height)

  for _,plot in pairs(self.plots) do
    if #plot.points > 2 then
      r,g,b,a = plot.line_color[1], plot.line_color[2], plot.line_color[3], plot.line_color[4] or 1
      love.graphics.setColor(r,g,b,a)

      local pts = {}
      for i=1,#plot.points/2 do
        pts[2*i-1]=affine_map(plot.points[2*i-1], self.lower_x_range, self.upper_x_range, 0, width)
        pts[2*i]=affine_map(plot.points[2*i], self.lower_y_range, self.upper_y_range, height, 0)

        --print(pts[2*i-1], pts[2*i])
      end
      --while true do end

      love.graphics.setLineWidth(plot.line_thickness)
      love.graphics.line(pts)
    end
  end
  love.graphics.setColor(1,1,1,1)

  love.graphics.setCanvas()
  love.graphics.draw(canvas, x, y)
  canvas = nil

  ---------------------------------------
  -- drawing x and y labels and tick marks
  local tick_number_spacing = 10

  local dx,dy=0,0
  if self.show_tick_marks then dx = 30; dy=20 end

  -- x axis ticks
  local tick_x = self.x_tick_marks_spacing*math.ceil(self.lower_x_range/self.x_tick_marks_spacing)
  love.graphics.setLineWidth(self.tick_mark_thickness)
  while tick_x <= self.upper_x_range do
    local tx = x+affine_map(tick_x, self.lower_x_range, self.upper_x_range, 0, width)
    love.graphics.line(tx, y+height-self.tick_mark_size/2, tx, y+height+self.tick_mark_size/2)

    -- number
    local tick = love.graphics.newText(self.label_font, tick_x)--string.format("%".. self.tick_mark_number_decimal_places.. "f", tick_x))
    local tw = tick:getWidth()
    local th = tick:getHeight()
    love.graphics.draw(tick, tx-tw/2, y+height+self.tick_mark_thickness+tick_number_spacing)

    tick_x = tick_x + self.x_tick_marks_spacing
  end

  -- y axis ticks
  local tick_y = self.y_tick_marks_spacing*(math.ceil(self.lower_y_range/self.y_tick_marks_spacing))
  love.graphics.setLineWidth(self.tick_mark_thickness)
  while tick_y <= self.upper_y_range do
    local ty = y+affine_map(tick_y, self.lower_y_range, self.upper_y_range, height, 0)
    love.graphics.line(x-self.tick_mark_size/2, ty, x+self.tick_mark_size/2, ty)

    -- number
    local tick = love.graphics.newText(self.label_font, tick_y)--string.format("%".. self.tick_mark_number_decimal_places.. "f", tick_x))
    local tw = tick:getWidth()
    local th = tick:getHeight()
    love.graphics.draw(tick, x-self.tick_mark_thickness-tick_number_spacing-tw, ty-th/2)

    tick_y = tick_y + self.y_tick_marks_spacing
  end


  --drawing axis labels
  x_label = love.graphics.newText(self.label_font, self.x_label)
  x_w = x_label:getWidth()
  love.graphics.draw(x_label, x+width-x_w, y+height+self.margin+dy)


  love.graphics.push()
  y_label = love.graphics.newText(self.label_font, self.y_label)
  y_w = y_label:getWidth()
  y_h = y_label:getHeight()
  love.graphics.translate(x-self.margin-y_h-dx, y+y_w)
  love.graphics.rotate(-math.pi/2)

  love.graphics.draw(y_label, 0, 0)
  love.graphics.pop()
end



return grapher
