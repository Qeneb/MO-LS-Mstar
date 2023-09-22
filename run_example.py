import time
import json
import numpy as np
import matplotlib.pyplot as plt

import libmomapf.common as common
import libmomapf.mocbs as mocbs
import libmomapf.moastar as moastar # NAMOA*
import libmomapf.momstar as momstar
import libmomapf.molss as molss
import demo
import random
import math
import maploader
random.seed(1)

  
def run_room_32():
  # grids = maploader.load('data/room-32-32-4.map/room-32-32-4.map')
  grids = np.zeros((32,32))
  # sy = np.array([52,17,14,17,44,29,6,33,6,38]) # start y = rows in grid image, the kth component corresponds to the kth robot.
  # sx = np.array([61,63,60,23,44,23,41,4,62,13]) # start x = column in grid image
  # gy = np.array([54,20,12,19,46,27,7,35,6,37]) # goal y
  # gx = np.array([61,61,59,21,46,23,43,3,60,14]) # goal x

  # sy = np.array([52,59]) # start y = rows in grid image, the kth component corresponds to the kth robot.
  # sx = np.array([30,29]) # start x = column in grid image
  # gy = np.array([61,51]) # goal y
  # gx = np.array([31,30]) # goal x

  # cvecs = [np.array([2,2]), np.array([2,3])] # the kth component corresponds to the kth robot.
  # cgrids = [np.ones((64,64)), np.ones((64,64))] # the mth component corresponds to the mth objective.
  scen_num = 2
  robot_num = 2
  # sx = np.array([0,1,17,18,25,18,20])
  # sy = np.array([26,27,6,26,19,16,21])
  # gx = np.array([0,2,17,26,31,19,27])
  # gy = np.array([17,17,1,25,21,6,19])
  sx,sy,gx,gy = maploader.scenloader('data/room-32-32-4.map/scen-even/room-32-32-4-even-1.scen', scen_num, robot_num)
  
  # cvecs = [np.array([2,2]), np.array([1,5]), np.array([3,4]), np.array([4,1]),\
  #   np.array([5,1/2]), np.array([1,20]), np.array([2,3]), np.array([4,2]),\
  #   np.array([2,1]), np.array([1,4])] # the kth component corresponds to the kth robot.
  cvecs = []
  cdims = 2
  for i in range(robot_num):
    vec = []
    for j in range(cdims):
      vec.append(random.randint(1,2))
    cvecs.append(np.array(vec))
  # cgrids = [np.random.randint(1, 11, size=(64, 64)),np.random.randint(1, 11, size=(64, 64))]
  cgrids = [np.random.randint(1, 11, size=(32, 32)) for i in range(robot_num)] # the mth component corresponds to the mth objective.
  # cost for agent-i to go through an edge c[m] = cvecs[i][m] * cgrids[m][vy,vx], where vx,vy are the target node of the edge.
  
  cdim = len(cvecs[0])

  res = molss.RunMoLSSMstarMAPF(grids, sx, sy, gx, gy, cvecs, cgrids, cdim, 1.0, 0.0, np.inf, 180)
  # print(res)
  with open("exp9_"+"scen"+str(scen_num)+"_sols.json", "w") as f:
    json.dump(res[0], f, indent=2, ensure_ascii=False)
  with open("exp9_"+"scen"+str(scen_num)+"_objs.json", "w") as f:
    json.dump(res[1][1], f, indent=2, ensure_ascii=False)

  with open("exp9_"+"scen"+str(scen_num)+"_paras.json", "w") as f:
    f.write(str(res[1][0]))
    f.write('\n')
    f.write(str(res[1][2]))
    f.write('\n')
    f.write(str(res[1][3]))
    f.write('\n')
    f.write(str(res[1][4]))
    f.write('\n')
    f.write(str(res[1][5]))
    f.write('\n')
    
  # plt.rcParams['font.size'] = 60  # 设置字体大小
  # plt.rcParams['figure.figsize'] = (30, 30)
  # for id in res[0]:
  #   sol = res[0][id]
  #   fig, ax = plt.subplots()
  #   demo.draw_grid_map(ax, grids)
  #   demo.draw_one_solution(fig, ax, sol, 64, 'test' + str(id) + '_scen'+str(scen_num)+'.mp4')

  return

def run_virtual_map():
  grids = np.zeros((41,21))
  sy = np.array([0,10]) # start y = rows in grid image, the kth component corresponds to the kth robot.
  sx = np.array([0,10]) # start x = column in grid image
  gy = np.array([1,10]) # goal y
  gx = np.array([1,11]) # goal x
  cvecs = [np.array([2,2,1]), np.array([1,5,1])] # the kth component corresponds to the kth robot.
  cgrids = [np.ones((41,21)), np.ones((41,21)), np.ones((41,21))] # the mth component corresponds to the mth objective.
  # cost for agent-i to go through an edge c[m] = cvecs[i][m] * cgrids[m][vy,vx], where vx,vy are the target node of the edge.

  cdim = 3
  res = molss.RunMoLSSMstarMAPF(grids, sx, sy, gx, gy, cvecs, cgrids, cdim, 1.0, 0.0, np.inf, 60)
  print(res)
  with open("exp10_"+"sols.json", "w") as f:
    json.dump(res[0], f, indent=2, ensure_ascii=False)
  with open("exp10_"+"objs.json", "w") as f:
    json.dump(res[1][1], f, indent=2, ensure_ascii=False)
  with open("exp10_"+"paras.json", "w") as f:
    f.write(str(res[1][0]))
    f.write('\n')
    f.write(str(res[1][2]))
    f.write('\n')
    f.write(str(res[1][3]))
    f.write('\n')
    f.write(str(res[1][4]))
    f.write('\n')
    f.write(str(res[1][5]))
    f.write('\n')

def main():
  run_room_32()
  # run_virtual_map()
  return

if __name__ == '__main__':
  print("begin of main")
  # random.seed(1)
  main()
  print("end of main")
