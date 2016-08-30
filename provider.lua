require 'nn'
require 'image'
require 'xlua'
require 'csvigo'

local Provider = torch.class 'Provider'

function Provider:__init(full)
  local trsize = 50
  local tesize = 450


  -- load dataset
  self.trainData = {
     data = torch.Tensor(450, 14040),
     labels = torch.Tensor(450),
     size = function() return trsize end
  }
	for i = 1, 450 do
		if subset == nil then
            --resize image to 30x39. Note torch.reshape takes in height then width, opp of image.scale
			subset = torch.reshape(image.scale(image.load('./Data/SCUT-FBP-'.. i ..'.jpg',3,'byte'), 60, 78), 1, 3,78,60)	
		else
			subset = torch.cat(subset, torch.reshape(image.scale(image.load('./Data/SCUT-FBP-'.. i ..'.jpg',3,'byte'), 60, 78), 1, 3,78,60), 1)
		end
	end

    for i = 451, 500 do
		if subset_test == nil then
			subset_test = torch.reshape(image.scale(image.load('./Data/SCUT-FBP-'.. i ..'.jpg',3,'byte'), 60, 78) , 1, 3,78,60)	
		else
			subset_test = torch.cat(subset_test, torch.reshape(image.scale(image.load('./Data/SCUT-FBP-'.. i ..'.jpg',3,'byte'), 60, 78), 1, 3,78,60), 1)
		end
	end

    local labels = csvigo.load("Attractiveness label.csv")
    labels = torch.Tensor(labels.score)
    
    train_Y = labels[{{1,450}}] 
    test_Y = labels[{{451,500}}]  



  local trainData = self.trainData
  --print(#subset.train_X)
  --divide by 255 to make pixel [0,1]
  trainData.data = subset:double() / 255
  --print(#trainData.data)
  trainData.labels = train_Y

  self.testData = {
     data = subset_test:double() / 25,
     labels =test_Y:double(),
     size = function() return tesize end
  }
  local testData = self.testData
  --testData.labels = testData.labels + 1
  --image.save("train.png", trainData.data[2])
  --image.save("train2.png", image.scale(image.load('./Data/SCUT-FBP-'.. 1 ..'.jpg',3,'byte'),60,78))
end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
  local trainData = self.trainData
  local testData = self.testData

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     print(#yuv)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v

  -- preprocess testSet
  for i = 1,testData:size() do
    xlua.progress(i, testData:size())
     -- rgb -> yuv
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
  end
  -- normalize u globally:
  testData.data:select(2,2):add(-mean_u)
  testData.data:select(2,2):div(std_u)
  -- normalize v globally:
  testData.data:select(2,3):add(-mean_v)
  testData.data:select(2,3):div(std_v)
end

