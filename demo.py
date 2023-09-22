import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import math
import json
import maploader

circle_list = []
text_list = []
flag = True

def draw_grid_map(ax, grid):
    plt.imshow(grid,cmap='binary')

def draw_one_agent(ax, x, y, k=0, color='r'):
    circle = plt.Circle((x, y), 0.4, facecolor=color, edgecolor='black',linewidth = 1)
    text = plt.text(x, y, str(k),fontdict={'fontsize':1})
    ax.add_artist(circle)
    ax.add_artist(text)
    return circle, text
    

def transfer_id_into_cor(id, n):
    x = id%n
    y = math.floor(id/n)
    return x, y

def one_solution_update(i, ax, sol, n):
    global circle_list, text_list, flag
    nr = len(sol[0][0])
    
    if i == 0 and flag: 
        for k in range(nr):
            x,y = transfer_id_into_cor(sol[0][1][k],n)
            circle, text = draw_one_agent(ax, x, y, k, 'r')
            circle_list.append(circle)
            text_list.append(text)
        flag = False
    else:
        for m in range(len(sol)-1):
            t1 = sol[m][0]
            t2 = sol[m+1][0]
            loc1 = sol[m][1]
            loc2 = sol[m+1][1]
            for j in range(len(t1)):
                step = math.floor(i/5)
                if step >= t1[j] and step < t2[j]:
                    x1, y1 = transfer_id_into_cor(loc1[j],n)
                    x2, y2 = transfer_id_into_cor(loc2[j],n)
                    x = (i - 5*t1[j])/(5 * (t2[j]-t1[j])) * (x2 - x1) + x1
                    y = (i - 5*t1[j])/(5 * (t2[j]-t1[j])) * (y2 - y1) + y1
                    circle_list[j].center = (x, y)
                    text_list[j].set_position((x, y))

def draw_one_solution(fig, ax, sol, n, filename):
    global circle_list,text_list,flag
    b = [i[0] for i in sol]
    c = [[b[i][j] for i in range(len(b))] for j in range(len(b[0]))]
    frames = 5*(int(max([max(row) for row in c])) + 1)
    ani = FuncAnimation(fig, one_solution_update, frames=frames, interval=33, fargs=(ax, sol, n))
    ani.save(filename, writer='ffmpeg')
    circle_list = []
    text_list = []
    flag = True
    fig.clf()

def draw_animation(fname,scen_num,grids):
    with open(fname) as f:
        sols = json.load(f)
    plt.rcParams['font.size'] = 60  # 设置字体大小
    plt.rcParams['figure.figsize'] = (30, 30)
    for id in sols:
        sol = sols[id]
        fig, ax = plt.subplots()
        draw_grid_map(ax, grids)
        draw_one_solution(fig, ax, sol, len(grids), 'test' + str(id) + '_scen'+str(scen_num)+'.mp4')


if __name__ == '__main__':
    
    # grids = np.zeros((3,3))
    # grids[0,:]=1
    # grids[1,0]=1
    # grids[1,2]=1

    # plt.rcParams['font.size'] = 30  # 设置字体大小
    # plt.rcParams['figure.figsize'] = (30, 30)
    # fig, ax = plt.subplots()
    # draw_grid_map(ax, grids)
    # a = [(np.array([0.0,0.0]),(6,8)), (np.array([2.0,1.0]),(7,8)),\
    #     (np.array([2.0,2.0]),(7,8)), (np.array([4.0,3.0]),(4,8)),\
    #     (np.array([4.0,4.0]),(4,7)), (np.array([6.0,5.0]),(7,6)),\
    #     (np.array([8.0,5.0]),(8,6))]
    # b = [i[0] for i in a]
    # c = [[b[i][j] for i in range(len(b))] for j in range(len(b[0]))]
    # frames = 10*(int(max([max(row) for row in c])) + 1)
    # print(frames)
    # ani = FuncAnimation(fig, one_solution_update, frames=frames, interval=100, fargs=(ax, a))
    # ani.save('test1.mp4',writer='ffmpeg')
    # plt.show()
    grids = maploader.load('data/room-32-32-4.map/room-32-32-4.map')
    draw_animation('exp8_scen2_sols.json',0,grids)
