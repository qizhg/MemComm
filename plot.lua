
require'gnuplot'

local f = torch.load('eps/mem10epsdatatotal')
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

x10 = torch.rand(epochs)
for n = 1, epochs do
	x10[n]=n
end

y10 = torch.rand(epochs,num_of_experiments)
for i = 1, num_of_experiments do
	for n = 1, epochs do
		y10[n][i] = g_logs[i][n].success
	end
end

y10_mean = torch.mean(y10,2)
y10_err = torch.std(y10,2) / torch.sqrt(num_of_experiments)
y10_mean = torch.squeeze(y10_mean)
y10_err = torch.squeeze(y10_err)
y10_high = y10_mean + y10_err
y10_low = y10_mean - y10_err
yy10 = torch.cat(x10,y10_low,2)
yy10 = torch.cat(yy10,y10_high,2)



gnuplot.pngfigure('11by11.png')
gnuplot.plot(
	{yy1,'with filledcurves fill transparent solid 0.2 ls 1'},
	{'memory size=10,eps',x1,y1_mean,'with lines ls 1'},
	{yy10,'with filledcurves fill transparent solid 0.2 ls 3'},
	{'memory size=10',x10,y10_mean,'with lines ls 3'}
	)
gnuplot.xlabel('epochs(1 epoch = 500 episodes)')
gnuplot.ylabel('success rate')
gnuplot.plotflush()
--]]
