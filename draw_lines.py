import matplotlib.pyplot as plt
# 拿数据
epoch_num = []
small_object_map = []
data = open("results20211106-185354.txt")
lines = data.readlines()
for line in lines:
  epoch, data = line.split(" ", 1)
  epoch_data = int(epoch.split(":", 1)[1])
  samll_object_data = float(data.split("  ", 13)[3])
  epoch_num.append(epoch_data)
  small_object_map.append(samll_object_data)

# 画图
# plot the real data
print(epoch_num)
print(small_object_map)


fig = plt.figure(figsize=(24, 12))
#plt.subplot(121)
plt.plot(epoch_num, small_object_map, color='r', linestyle='-')
#plt.subplot(122)
#plt.plot(epoch_num, small_object_map, color='r', linestyle='--')
plt.show()