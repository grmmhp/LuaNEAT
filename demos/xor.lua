local LuaNEAT = require"LuaNEAT"

local training_set = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0}
}

local size = 100
local inputs,outputs=2,1
local pool = LuaNEAT.newPool(size, inputs, outputs)
pool:initialize()

local function stats(net)
  local sum=0
  for n=1,#training_set do
    local outputs = net:forward("active", training_set[n][1], training_set[n][2])
    print("inputs: (".. training_set[n][1] .. ", ".. training_set[n][2] .. "); output: ".. outputs[1])
    local err = (outputs[1] - training_set[n][3])^2
    sum = sum + err
  end
  print("fit ".. 1-sum)
end


local max_gen = 100

print("\ngen 0")
stats(pool.nets[1])

for gen=1,max_gen do
  local nets = pool:getNeuralNetworks()
  for i=1,#nets do
    local net = nets[i]

    local sum=0
    for n=1,#training_set do
      outputs = net:forward("active", training_set[n][1], training_set[n][2])
      local err = (outputs[1] - training_set[n][3])^2
      sum = sum + err
    end
    net:setFitness(math.max(0, 1-sum))
  end

  local st = pool:nextGeneration()
  print(st)
  print("best fit ".. pool.last_best.fitness .."\n")
end

print("\nlast gen")
stats(pool.last_best:buildNeuralNetwork())
