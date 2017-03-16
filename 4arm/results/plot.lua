
require'gnuplot'

local f = torch.load('mem10epsdata')
g_logs = f.log
epochs = #g_logs[1]

num_of_experiments = #g_logs
x1 = torch.rand(epochs)
for n = 1, epochs do
	x1[n]=n
end

y1 = torch.rand(epochs,num_of_experiments)
for i = 1, num_of_experiments do
	for n = 1, epochs do
		y1[n][i] = g_logs[i][n].success
	end
end

y1_mean = torch.mean(y1,2)
y1_err = torch.std(y1,2) / torch.sqrt(num_of_experiments)
y1_mean = torch.squeeze(y1_mean)
y1_err = torch.squeeze(y1_err)
y1_high = y1_mean + y1_err
y1_low = y1_mean - y1_err
yy1 = torch.cat(x1,y1_low,2)
yy1 = torch.cat(yy1,y1_high,2)

-------------------------------------------------
local f = torch.load('4arm/mem10data')
g_logs = f.log
epochs = #g_logs[1]

num_of_experiments = #g_logs

x2 = torch.rand(epochs)
for n = 1, epochs do
	x2[n]=n
end

y2 = torch.rand(epochs,num_of_experiments)
for i = 1, num_of_experiments do
	for n = 1, epochs do
		y2[n][i] = g_logs[i][n].success
	end
end

y2_mean = torch.mean(y2,2)
y2_err = torch.std(y2,2) / torch.sqrt(num_of_experiments)
y2_mean = torch.squeeze(y2_mean)
y2_err = torch.squeeze(y2_err)
y2_high = y2_mean + y2_err
y2_low = y2_mean - y2_err
yy2 = torch.cat(x2,y2_low,2)
yy2 = torch.cat(yy2,y2_high,2)
-------------------------------------------------
local f = torch.load('4arm/mem0data')
g_logs = f.log
epochs = #g_logs[1]

num_of_experiments = #g_logs

x3 = torch.rand(epochs)
for n = 1, epochs do
	x3[n]=n
end

y3 = torch.rand(epochs,num_of_experiments)
for i = 1, num_of_experiments do
	for n = 1, epochs do
		y3[n][i] = g_logs[i][n].success
	end
end

y3_mean = torch.mean(y3,2)
y3_err = torch.std(y3,2) / torch.sqrt(num_of_experiments)
y3_mean = torch.squeeze(y3_mean)
y3_err = torch.squeeze(y3_err)
y3_high = y3_mean + y3_err
y3_low = y3_mean - y3_err
yy3 = torch.cat(x3,y3_low,2)
yy3 = torch.cat(yy3,y3_high,2)

---------------------------------------------------

gnuplot.pngfigure('11by11.png')
gnuplot.plot(
	{yy1,'with filledcurves fill transparent solid 0.2 ls 1'},
	{'memory size=10,A3C w/ e-greedy',x1,y1_mean,'with lines ls 1'},
	{yy2,'with filledcurves fill transparent solid 0.2 ls 2'},
	{'memory size=10,A3C',x2,y2_mean,'with lines ls 2'},
	{yy3,'with filledcurves fill transparent solid 0.2 ls 3'},
	{'memory size= 0, A3C',x3,y3_mean,'with lines ls 3'}
	)
gnuplot.xlabel('epochs(1 epoch = 500 episodes)')
gnuplot.ylabel('success rate')
gnuplot.plotflush()
--]]
