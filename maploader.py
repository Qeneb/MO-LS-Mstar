import json
import numpy as np

def load(filename1):
    #输入加载的文件名，输出包含地图信息的字典
    f = open(filename1,'r') #r表示只读模式
    map = []
    for line in f:
        if line.find('type ')>=0:
            type = line.split()[1]
        else: 
            if line.find('height ')>=0:
                height = int(line.split()[1])
            else:
                if line.find('width ')>=0:
                    width = int(line.split()[1])
                else:
                    if line != 'map\n':
                        row = []
                        for char in line:
                            if char == '.':
                                row.append(0) 
                            if char == '@':
                                row.append(1)
                        map.append(row)
    return np.array(map)

def scenloader(fname, bucket_num, robots_num):
    sx = []
    sy = []
    gx = []
    gy = []
    f = open(fname,'r')
    f.readline()
    for line in f:
        line_list = line.split('\t')
        bucket_id = line_list[0]
        if bucket_id == str(bucket_num):
            sx.append(int(line_list[4]))
            sy.append(int(line_list[5]))
            gx.append(int(line_list[6]))
            gy.append(int(line_list[7]))
            if len(sx) == robots_num:
                break
    return np.array(sx),np.array(sy),np.array(gx),np.array(gy)

if __name__ == '__main__':
    scenloader('data/room-64-64-8.map/scen-even/room-64-64-8-even-1.scen', 1)