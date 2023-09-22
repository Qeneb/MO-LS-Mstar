import numpy as np
import itertools as itt
import time
import copy
import libmomapf.moastar as moastar 
import libmomapf.loadmap
import libmomapf.common as cm

MAX_NGH_SIZE=1e7

class MoLSSState:
  """
  By dhk
  """
  def __init__(self, sid=-1, locs=(), G=np.mat(0), plocs=()):
    self.id = sid
    self.locs = locs # location ids ((agent1 loc),(agent2 loc)...... (agent3 loc))
    self.G = G
    self.plocs = plocs
  def __str__(self):
    return "{"+str(self.id)+","+str(self.locs)+","+str(self.G)+","+str(self.plocs)+"}"
  def ConflictSet(self):
    """
    Should only be called for MAPF.
    Only vertex conflicts
    TODO: edge conflicts
    """
    out_dict = dict()
    # 遍历所有Agents Pair Cn2组合问题
    for ix in range(len(self.locs)):
      for iy in range(ix+1, len(self.locs)):
        # vertex conflicts
        if self.locs[ix] == self.locs[iy]: # occupy same node
          if ix not in out_dict:
            if iy not in out_dict: # both not in dic
              out_dict[ix] = iy
              out_dict[iy] = iy
            else: # iy in dic, i.e. in some col set
              out_dict[ix] = cm.UFFind(out_dict, iy)
          else:
            # ix in dic, i.e. in some col set
            if iy not in out_dict:
              out_dict[iy] = cm.UFFind(out_dict, ix)
            else: # both in dict
              cm.UFUnion(out_dict, ix, iy)
    return out_dict

class MoLSSMAPFBase:
  """
  MoLSSMAPF, no heuristic.
  This class is a base class.
  """
  def __init__(self, grids, sx, sy, gx, gy, cvecs, cost_grids, w, eps, time_limit):
    """
    NAMOA* algorithm.
    cvecs e.g. = [np.array(1,2),np.array(3,4),np.array(1,5)] means 
      robot 1 has cost (1,2) over every edge, 
      robot 2 has cost (3,4) over every edge and 
      robot 3 has cost (1,5) over every edge.
    """

    # shape = (21,41)
    self.grids = grids 
    (self.nyt, self.nxt) = grids.shape
    self.cost_vecs = copy.deepcopy(cvecs)
    self.cdim = len(cvecs[0])
    self.num_robots = len(sx)
    self.state_gen_id = 3 # 1 is start, 2 is goal
    self.cost_grids = copy.deepcopy(cost_grids)

    fnames = ["data/1-c1.gr","data/1-c2.gr","data/1-c3.gr"]
    self.edge_cost_dict = libmomapf.loadmap.read_roadmap_from_file(fnames)

    # start state
    self.sx = copy.deepcopy(sx)
    self.sy = copy.deepcopy(sy)
    tmp_sloc = list()
    for idx in range(len(self.sx)):
      tmp_sloc.append( self.sy[idx]*self.nxt + self.sx[idx] )
    self.s_o = MoLSSState(1,tuple(tmp_sloc),np.zeros((self.cdim, self.num_robots)),tuple(tmp_sloc))

    # goal state
    self.gx = copy.deepcopy(gx)
    self.gy = copy.deepcopy(gy)
    tmp_gloc = list()
    for idx in range(len(self.gx)):
      tmp_gloc.append( self.gy[idx]*self.nxt + self.gx[idx] )
    self.s_f = MoLSSState(2,tuple(tmp_gloc),np.full((self.cdim, self.num_robots),np.inf),tuple(tmp_gloc))

    # search params and data structures
    self.weight = w
    self.eps = eps
    self.action_set_x = [0,-1,0,1,0]
    self.action_set_y = [0,0,-1,0,1]
    self.action_set_even_x = [-1,-1,0,0,+1,+1]
    self.action_set_even_y = [0,+1,-1,+1,0,+1]
    self.action_set_odd_x = [-1,-1,0,0,+1,+1]
    self.action_set_odd_y = [-1,0,-1,+1,-1,0]
    self.all_visited_s = dict()
    self.frontier_map = dict()
    self.open_list = cm.PrioritySet()
    self.f_value = dict()
    self.close_set = set()
    self.backtrack_dict = dict() # track parents

    self.time_limit = time_limit
    self.remain_time = time_limit # record remaining time for search
    self.heu_failed = False # by default

    return

  def GetRemainTime(self):
    return self.remain_time

  def GetHeuristic(self, s):
    """
    Get estimated cdim X num_robots -dimensional cost-to-goal.
    In this base class, there is no heuristic, just zero h-matrix...
    """
    return np.zeros((self.cdim, self.num_robots))

  def GetCost(self, loc, nloc):
    """
    Get cdim X num_robots -dimenisonal cost vector of moving from loc to nloc.
    """
    out_cost = np.zeros((self.cdim, self.num_robots))
    for idx in range(self.num_robots): # loop over all robots
      if nloc[idx] != loc[idx]: # there is a move, not wait in place.
        if len(self.cost_grids) > 0 and len(self.cost_grids) >= self.cdim :
          cy = int(np.floor(nloc[idx]/self.nxt)) # ref x
          cx = int(nloc[idx]%self.nxt) # ref y, 
          # ! CAVEAT, this loc is ralavant to cost_grid, must be consistent with Policy search.
          for ic in range(self.cdim):
            out_cost[ic][idx] = self.cost_vecs[idx][ic]*self.cost_grids[ic][cy,cx]
        else:
          out_cost = out_cost + self.cost_vecs[idx]
      else: # robot stay in place.
        pass
        # if len(self.s_f.locs) == 0: # policy mode, no goal specified.
        #   out_cost = out_cost + self.cost_vecs[idx]*1 # stay-in-place fixed cost
        # elif loc[idx] != self.s_f.locs[idx]: # nloc[idx] = loc[idx] != s_f.loc[idx], robot not reach goal.
        #   out_cost[:,idx] = out_cost[:,idx] + self.cost_vecs[idx]*1 # np.ones((self.cdim)) # stay in place, fixed energy cost for every robot
    return out_cost

  def AddToFrontier(self, s):
    """Add a state into frontier"""
    self.all_visited_s[s.id] = s
    if s.locs not in self.frontier_map:
      self.frontier_map[s.locs] = set()
      self.frontier_map[s.locs].add(s.id)
    else:
      self.RefineFrontier(s)
      self.frontier_map[s.locs].add(s.id)
    return

  def RefineFrontier(self, s):
    """Use s to remove dominated states in frontier set"""
    
    if s.locs not in self.frontier_map:
      return
    temp_frontier = copy.deepcopy(self.frontier_map[s.locs])
    for sid in temp_frontier:
      if sid == s.id: # do not compare with itself !
        continue
      if sid not in self.f_value:
        self.f_value[sid] = self.all_visited_s[sid].G + self.weight*self.GetHeuristic(self.all_visited_s[sid])
      if s.id not in self.f_value:
        self.f_value[s.id] = s.G + self.weight*self.GetHeuristic(s)
      # if cm.TimeDomOrEqual(s.G, self.all_visited_s[sid].G):
      if cm.TimeDomOrEqual(self.f_value[s.id], self.f_value[sid]):
        print("refine")
        self.frontier_map[s.locs].remove(sid)
        self.open_list.remove(sid)
    return

  def CollisionCheck(self,s,ns):
    out_dict = ns.ConflictSet() # conflict in ns
    # C n_robots 2_robots
    for idx in range(len(s.locs)):
      for idy in range(idx+1, len(s.locs)):
        conflict_flag = False
        if idx == idy:
          continue
        # robot idx from s.locs[idx] in timestamp s.G[0][idx] to ns.locs[idx] in timestamp ns.G[0][idx]
        # robot idy from s.locs[idy] in timestamp s.G[0][idy] to ns.locs[idy] in timestamp ns.G[0][idy]
        if (ns.locs[idx] == s.locs[idy] and s.locs[idx] == ns.locs[idy]) or (ns.locs[idx] == ns.locs[idy] and s.locs[idy] == s.locs[idx]):
        # Only two robots swap location or they start at the same location and end at the same location
        # can lead to conflicts
          if ns.locs[idx] == s.locs[idx]:
          # If robot idx waits, it occupies it's loc in time interval [s.G[0][idx], ns.G[0][idx]]
            if s.locs[idy] == s.locs[idx] and (s.G[0][idy] <= ns.G[0][idx]) and (s.G[0][idy] >= s.G[0][idx]):
              conflict_flag = True
            if ns.locs[idy] == s.locs[idx] and (ns.G[0][idy] <= ns.G[0][idx]) and (ns.G[0][idy] >= s.G[0][idx]):
              conflict_flag = True
          else:
          # If robot idx moves, it occupies its corresponding location in timestamp s.G[0][idx] and ns.G[0][idx]
          # and the edge between s.locs[idx] and ns.locs[idx] during to = s.G[0][idx] and tf = ns.G[0][idx]
            # Edge conflict
            if ((ns.locs[idy] == s.locs[idx] and s.locs[idy] == ns.locs[idx]) or\
              (s.locs[idx] == s.locs[idy] and ns.locs[idx] == ns.locs[idy])) and\
              (ns.G[0][idx] > s.G[0][idy] or ns.G[0][idy] > s.G[0][idx]):
              conflict_flag = True
            # Vertex conflict
            if (s.locs[idx] == s.locs[idy] and s.G[0][idx] == s.G[0][idy]) or\
              (s.locs[idx] == ns.locs[idy] and s.G[0][idx] == ns.G[0][idy]) or\
              (ns.locs[idy] == s.locs[idy] and ns.G[0][idx] == s.G[0][idy]) or\
              (ns.locs[idx] == ns.locs[idy] and ns.G[0][idx] == ns.G[0][idy]):
              conflict_flag = True
        if conflict_flag:
          if idx not in out_dict:
            if idy not in out_dict: # both not in dic
              out_dict[idx] = idy
              out_dict[idy] = idy
            else: # iy in dic, i.e. in some col set
              out_dict[idx] = cm.UFFind(out_dict, idy)
          else:
            if idy not in out_dict:
              out_dict[idy] = cm.UFFind(out_dict, idx)
            else: # both in dict
              cm.UFUnion(out_dict, idx, idy)
    return out_dict

  def FilterState(self,s,f_array):
    if self.FrontierFilterState(s,f_array):
      return True
    if self.GoalFilterState(s,f_array):
      return True
    return False

  def FrontierFilterState(self,s,f_array):
    """
    filter state s, if s is dominated by any states in frontier other than s.
    """
    if s.locs not in self.frontier_map: # goal not reached yet.
      return False
    for sid in self.frontier_map[s.locs]:
      if sid == s.id:
        continue # do not compare with itself...
      if sid not in self.f_value:
        self.f_value[sid] = self.all_visited_s[sid].G + self.weight*self.GetHeuristic(self.all_visited_s[sid])
      if s.id not in self.f_value:
        self.f_value[s.id] = s.G + self.weight*self.GetHeuristic(s)

      if cm.TimeDomOrEqual(self.f_value[sid], self.f_value[s.id]):
      # if cm.TimeDomOrEqual(self.all_visited_s[sid].G, s.G):
        return True # filtered
    return False # not filtered

  def GoalFilterState(self,s,f_array):
    """
    filter state s, if s is dominated by any states that reached goal. (non-negative cost vec).
    """
    if self.s_f.locs not in self.frontier_map: # goal not reached yet.
      return False
    for sid in self.frontier_map[self.s_f.locs]:
      if cm.TimeDomOrEqual(self.f_value[sid], f_array):
      # if cm.TimeDomOrEqual(self.f_value[sid], f_array):
        return True
        
      if sid not in self.f_value:
        self.f_value[sid] = self.all_visited_s[sid].G + self.weight*self.GetHeuristic(self.all_visited_s[sid])
      if s.id not in self.f_value:
        self.f_value[s.id] = s.G + self.weight*self.GetHeuristic(s)
      
      # if cm.TimeDomOrEqual(self.all_visited_s[sid].G, s.G):
      if cm.TimeDomOrEqual(self.f_value[sid], self.f_value[s.id]):
        return True # filtered
    return False # not filtered

  def GetNeighbors(self, s, tstart):
    """
    input a tjs state s, compute its neighboring states.
    output a list of states.
    """
    nid_dict = dict() # key is robot index, value is ngh(location id) for robot idx
    for idx in range(self.num_robots): # loop over all robot

      tnow = time.perf_counter() # check if timeout.
      if (int(tnow - tstart) > self.GetRemainTime()):
        print(" FAIL! timeout in get ngh! " )
        return [], False

      nid_dict[idx] = list() # neighbors for each robot idx

      # its nid_dict[idx] is empty list
      cy = int(np.floor(s.locs[idx]/self.nxt)) # current x
      cx = int(s.locs[idx]%self.nxt) # current y

      # explore all neighbors
      for action_idx in range(len(self.action_set_x)):
        nx = cx+self.action_set_x[action_idx] # next x
        ny = cy+self.action_set_y[action_idx] # next y 
        if not (ny < 0 or ny >= self.nyt or nx < 0 or nx >= self.nxt):
          # do not exceed border
          if self.grids[ny,nx] == 0: # not obstacle
            nid_dict[idx].append( ny*self.nxt+nx )
 
    # generate all joint neighbors s_ngh from nid_dict
    s_ngh = list()
    all_loc = list( itt.product(*(nid_dict[ky] for ky in sorted(nid_dict))) )

    for ida in range(len(all_loc)): # loop over all neighboring joint edges
      tnow = time.perf_counter()
      if (int(tnow - tstart) > self.GetRemainTime()):
        print(" FAIL! timeout in get ngh! " )
        return [], False
      # Get New G
      
      # s.cost_vec+self.GetCost(s.locs,all_loc[ida])
      new_G = s.G+self.GetCost(s.locs,all_loc[ida])
      ns = MoLSSState(self.state_gen_id, tuple(all_loc[ida]), new_G, s.locs )
      self.state_gen_id = self.state_gen_id + 1 
      s_ngh.append(ns)
    return s_ngh, True

  def Pruning(self, s, f_array):
    """
    So called "frontier check"
    Check Dominance
    """
    if s.locs not in self.frontier_map:
      return False # this joint edge is never visited before, should not prune
    flag = False
    # Check if synchronized state has been generated
    for fid in self.frontier_map[s.locs]: # loop over all states in frontier set.
      s_now = self.all_visited_s[fid]
      tmp = s_now.G[0]
      sorted_tmp = sorted(tmp)
      if sorted_tmp[0] == sorted_tmp[-1]:
        # Synchronized State has been generated
        flag = True
        break
    for fid in self.frontier_map[s.locs]: # loop over all states in frontier set.
      if fid == s.id:
        continue
      if fid not in self.f_value:
        self.f_value[fid] = self.all_visited_s[fid].G + self.weight*self.GetHeuristic(self.all_visited_s[fid])
      if s.id not in self.f_value:
        self.f_value[s.id] = s.G + self.weight*self.GetHeuristic(s)

      if flag:
        if cm.TimeDomOrEqual2(self.f_value[fid], self.f_value[s.id]):
        # if cm.TimeDomOrEqual2(self.all_visited_s[fid].G, s.G):
          return True # should be pruned
      else:
        # if (fid == 519):
        #   breakpoint()
        # if cm.TimeDomOrEqual(self.all_visited_s[fid].G, s.G):
        if cm.TimeDomOrEqual(self.f_value[fid], self.f_value[s.id]):
          return True # should be pruned
    # end of for
    return False # should not be prune

  def ReconstructPath(self, sid):
    """
    input state is the one that reached, 
    return a list of joint vertices in right order.
    """
    jpath = []
    while sid in self.backtrack_dict:
      tmpt = self.all_visited_s[sid].G[0]
      tmploc = self.all_visited_s[sid].locs
      jpath.append((tuple([int(i) for i in tmpt.tolist()]), tuple([int(tmploc[i]) for i in range(len(tmploc))])))
      sid = self.backtrack_dict[sid] 
    tmpt = self.all_visited_s[sid].G[0]
    tmploc = self.all_visited_s[sid].locs
    jpath.append((tuple([int(i) for i in tmpt.tolist()]), tuple([int(tmploc[i]) for i in range(len(tmploc))])))
    # reverse output path here.
    out = []
    for idx in range(len(jpath)):
      out.append(jpath[len(jpath)-1-idx])
    return out

  def ReconstructPathAll(self):
    jpath_all = dict()
    if self.s_f.locs not in self.frontier_map:
      return jpath_all # no solution found
    for gid in self.frontier_map[self.s_f.locs]:
      jpath_all[gid] = self.ReconstructPath(gid)
    return jpath_all

class MoLSSMAPF(MoLSSMAPFBase):
  """
  MoLSSMAPF, derived from MoLSSMAPFBase, heuristic used.
  This is the LSS algorithm used for comparison in MOMAPF problem.
  """
  def __init__(self, grids, sx, sy, gx, gy, cvecs, cost_grids, w, eps, time_limit, compute_pol):
    super(MoLSSMAPF, self).__init__(grids, sx, sy, gx, gy, cvecs, cost_grids, w, eps, time_limit)
    self.time_limit = time_limit
    self.remain_time = time_limit # record remaining time for search
    # optimal policy
    self.optm_policis = dict()
    self.optm_distmats = dict()
    self.heu_failed = False
    if compute_pol:
      # ix 机器人的编号
      for ix in range(len(self.sx)):
        self.optm_policis[ix], self.optm_distmats[ix], search_res = \
          moastar.GetMoMapfPolicy(self.grids, gx[ix], gy[ix], cvecs[ix], cost_grids, self.remain_time)
        self.remain_time = self.remain_time - search_res[-1]
        if search_res[2] == 0:
          # policy build failed....
          self.heu_failed = True
          break
        if self.remain_time <= 0:
          self.heu_failed = True
          break
  
  def GetHeuristic(self, s):
    """
    Get estimated cdim X num_robots -dimensional cost-to-goal, 
    override the func in MoLSSMAPFBase class.
    """
    if len(self.sx)==1:
      return np.zeros(1)
    h_vec = np.zeros((self.cdim, self.num_robots))
    for ri in range(self.num_robots):
      nid = s.locs[ri]
      cy = int(np.floor(nid/self.nxt)) # current x
      cx = int(nid%self.nxt) # current y
      dist_vec_list = self.optm_distmats[ri][cy][cx]
      if len(dist_vec_list) > 0:
        h_vec[:,ri] = np.min(np.stack(dist_vec_list),axis=0)
    return h_vec

class MoLSSMstar(MoLSSMAPF):
  """MoMstar is derived from MoAstarMAPF"""
  def __init__(self, grids, sx, sy, gx, gy, cvecs, cost_grids, w, eps, time_limit, compute_pol):
    super(MoLSSMstar, self).__init__(grids, sx, sy, gx, gy, cvecs, cost_grids, w, eps, time_limit, compute_pol)
    self.collision_dict = dict() 
    self.backprop_dict = dict() 
    self.max_col_set = 0

  def AddToBackSet(self, sid, nsid):
    """
    """
    if nsid not in self.backprop_dict:
      self.backprop_dict[nsid] = set()
    self.backprop_dict[nsid].add(sid)
    return

  def ReopenState(self, sid):
    """
    """
    self.open_list.add(np.sum(self.f_value[sid]), sid )
    return

  def UnionColSet(self, s, cset):
    """
    """
    for k in cset: # union
      self.collision_dict[s.id][k] = 1
    if len(self.collision_dict[s.id]) > self.max_col_set:
      self.max_col_set = len(self.collision_dict[s.id])
    return 

  def BackPropagation(self, s, cset):
    """
    collision set back propagation
    """
    if len(cset) > self.max_col_set:
      self.max_col_set = len(cset)
    if cm.IsDictSubset(cset, self.collision_dict[s.id]):
      return
    self.UnionColSet(s, cset)
    self.ReopenState(s.id)
    if s.id not in self.backprop_dict:
      return # reach init state
    for ps_id in self.backprop_dict[s.id]:
      self.BackPropagation(self.all_visited_s[ps_id], cset)

  def GetNgh(self, sk, tstart, minDwait=1):
    skt_dict = dict()
    # Don't consider the robots that have reached the final 
    delete_list = []
    I_f = set()
    for i in range(len(sk.G[0])):
      if sk.locs[i] != self.s_f.locs[i]:
        skt_dict[i] = sk.G[0][i]
      else:
        I_f.add(i)
    sorted_skt = sorted(skt_dict.items(), key=lambda x: x[1])
    tmin = min(skt_dict.values())
    if sorted_skt[0][1] == sorted_skt[-1][1]:
      tmin2 = tmin
    else:
      for j in sorted_skt:
        if j[1] != tmin:
          tmin2 = j[1]
          break

    I = set(range(len(sk.G[0])))
    Itmin = set([i for i in skt_dict if skt_dict[i] == tmin])
    
    Sngh = []
    nid_dict = dict()
    ngh_size = 1
    wait_dict = dict()
    for i in range(len(I)):
      tnow = time.perf_counter() # check if timeout.
      if (int(tnow - tstart) > self.GetRemainTime() ):
        print(" FAIL! timeout in get ngh! " )
        return [], False
      if (i not in Itmin) or (i in I_f):
        # A Tuple
        wait_dict[i] = 0
        nid_dict[i] = [sk.locs[i]]
      else:
        nid_dict[i] = []
        cy = int(np.floor(sk.locs[i]/self.nxt)) # current y
        cx = int(sk.locs[i]%self.nxt) # current x
        if i in self.collision_dict[sk.id]:
          Dwait = 0
          if tmin == tmin2:
            Dwait = minDwait
          else:
            Dwait = tmin2 - tmin
          if cy%2 != 1:
            tmp_action_set_x = self.action_set_odd_x
            tmp_action_set_y = self.action_set_odd_y
          else:
            tmp_action_set_x = self.action_set_even_x
            tmp_action_set_y = self.action_set_even_y
          ngh_size = ngh_size * len(tmp_action_set_x)
          for action_idx in range(len(tmp_action_set_x)):
            nx = cx+tmp_action_set_x[action_idx] # next x
            ny = cy+tmp_action_set_y[action_idx] # next y 
            if not (ny < 0 or ny >= self.nyt or nx < 0 or nx >= self.nxt):
              # do not exceed border
              nloc = nx*self.nxt + ny
              loc = sk.locs[i]
              if self.edge_cost_dict[(loc,nloc)][0] < 999: # not obstacle
                nid_dict[i].append(nloc)
                if nx == cx and ny == cy:
                  wait_dict[i] = Dwait
        else:
          ngh_size = ngh_size * len(self.optm_policis[i][cy][cx])
          cx_cy_policis = self.optm_policis[i][cy][cx]
          # print(cx_cy_policis)
          for next_xy in cx_cy_policis: 
            nx = next_xy[0] # follow the convention in GridPolicy in common.py
            ny = next_xy[1]
            nid_dict[i].append(nx*self.nxt+ny)
            # if nx == cx and ny == cy:
            #   wait_dict[i] = 0
    if ngh_size > MAX_NGH_SIZE: # too many ngh, doom to fail
      print(" !!! ngh_size too large:", ngh_size, " > ", MAX_NGH_SIZE, " !!!")
      return list(), False
    all_loc = list( itt.product(*(nid_dict[ky] for ky in sorted(nid_dict))) )
    for ida in range(len(all_loc)):
      tnow = time.perf_counter()
      if (int(tnow - tstart) > self.GetRemainTime() ):
        print(" FAIL! timeout in get ngh! " )
        return [], False
      # wait or don't plan are tuples: (id, wait_time)
      new_locs = tuple(all_loc[ida])
      new_G = sk.G + self.GetCost(sk.locs, all_loc[ida])
      for i in range(len(new_locs)):
        if new_locs[i] == sk.locs[i]:
          new_G[0][i] = new_G[0][i] + wait_dict[i]
      ns = MoLSSState(self.state_gen_id, new_locs, new_G, sk.locs)
      self.state_gen_id = self.state_gen_id + 1 
      Sngh.append(ns)
    return Sngh, True

  def GetNeighbors(self, s, tstart):
    """
    """
    nid_dict = dict()
    ngh_size = 1
    for idx in range(self.num_robots): # loop over all robot
      tnow = time.perf_counter() # check if timeout.
      if (int(tnow - tstart) > self.GetRemainTime() ):
        print(" FAIL! timeout in get ngh! " )
        return [], False
      nid_dict[idx] = list()
      cy = int(np.floor(s.locs[idx]/self.nxt)) # current y
      cx = int(s.loc[idx]%self.nxt) # current x
      if idx in self.collision_dict[s.id]:
        ngh_size = ngh_size * len(self.action_set_x)
        for action_idx in range(len(self.action_set_x)):
          nx = cx+self.action_set_x[action_idx] # next x
          ny = cy+self.action_set_y[action_idx] # next y 
          if not (ny < 0 or ny >= self.nyt or nx < 0 or nx >= self.nxt):
            # do not exceed border
            if self.grids[ny,nx] == 0: # not obstacle
              nid_dict[idx].append( ny*self.nxt+nx )
      else:
        ngh_size = ngh_size * len(self.optm_policis[idx][cy][cx])
        for next_xy in self.optm_policis[idx][cy][cx]: 
          nx = next_xy[0] # follow the convention in GridPolicy in common.py
          ny = next_xy[1]
          nid_dict[idx].append( ny*self.nxt+nx )
    if ngh_size > MAX_NGH_SIZE: # too many ngh, doom to fail
      print(" !!! ngh_size too large:", ngh_size, " > ", MAX_NGH_SIZE, " !!!")
      return list(), False
    s_ngh = list()
    all_loc = list( itt.product(*(nid_dict[ky] for ky in sorted(nid_dict))) )
    for ida in range(len(all_loc)):
      tnow = time.perf_counter()
      if (int(tnow - tstart) > self.GetRemainTime() ):
        print(" FAIL! timeout in get ngh! " )
        return [], False
      ns = moastar.MoAstarState(self.state_gen_id, tuple(all_loc[ida]), s.cost_vec+self.GetCost(s.loc,all_loc[ida]) )
      self.state_gen_id = self.state_gen_id + 1 
      s_ngh.append(ns)
    return s_ngh, True

  def FilterState(self,s,f_array):
    """
    """
    if self.s_f.locs not in self.frontier_map: # goal not reached yet.
      return False
    for sid in self.frontier_map[self.s_f.locs]: # notice: only states that reaches goal !!
      if sid == s.id:
        continue

      if sid not in self.f_value:
        self.f_value[sid] = self.all_visited_s[sid].G + self.weight*self.GetHeuristic(self.all_visited_s[sid])
      if s.id not in self.f_value:
        self.f_value[s.id] = s.G + self.weight*self.GetHeuristic(s)

      if cm.TimeDomOrEqual( self.f_value[sid], f_array ):
        return True # filtered
      if cm.TimeDomOrEqual(self.f_value[sid], self.f_value[s.id]):
      # if cm.TimeDomOrEqual(self.all_visited_s[sid].G, s.G):
        return True # filtered
    return False # not filtered

  def RefineFrontier(self, s):
    """
    A detail. MOM*, no refinement, otherwise need to properly maintain back_set.
    Use s to remove dominated states in frontier set
    when one dominates state sid in frontier set is removed, elements in back_set(sid) should 
    be added in back_set(s.id)
    """
    if s.locs not in self.frontier_map:
      return
    temp_frontier = copy.deepcopy(self.frontier_map[s.locs])
    for sid in temp_frontier:
      if sid == s.id: # do not compare with itself !
        continue
      
      if sid not in self.f_value:
        self.f_value[sid] = self.all_visited_s[sid].G + self.weight*self.GetHeuristic(self.all_visited_s[sid])
      if s.id not in self.f_value:
        self.f_value[s.id] = s.G + self.weight*self.GetHeuristic(s)


      # if cm.TimeDomOrEqual(s.G, self.all_visited_s[sid].G):
      if cm.TimeDomOrEqual(self.f_value[sid], self.f_value[s.id]):
        self.frontier_map[s.locs].remove(sid)
        self.open_list.remove(sid)
        parent_sid = self.backtrack_dict[sid]
        self.AddToBackSet(parent_sid, s.id)
    return

  def RefineGoalFrontier(self, s): 
    """
    Use s to remove dominated states in frontier set at goal.
    """
    if s.locs != self.s_f.locs:
      return
    if self.s_f.locs not in self.frontier_map:
      return
    temp_frontier = copy.deepcopy(self.frontier_map[self.s_f.locs])
    for sid in temp_frontier:
      if cm.MixedDominantLess(s.G, self.all_visited_s[sid].G):
        self.frontier_map[self.s_f.locs].remove(sid)
    return

  def DominanceBackprop(self, s):
    """
    s is dominated and pruned. backprop s relavant info.
    """
    if s.locs not in self.frontier_map:
      return
    if s.id not in self.backtrack_dict:
      return
    parent_sid = self.backtrack_dict[s.id]
    for sid in self.frontier_map[s.locs]:

      if sid not in self.f_value:
        self.f_value[sid] = self.all_visited_s[sid].G + self.weight*self.GetHeuristic(self.all_visited_s[sid])
      if s.id not in self.f_value:
        self.f_value[s.id] = s.G + self.weight*self.GetHeuristic(s)

      # if cm.TimeDomOrEqual(self.all_visited_s[sid].G, s.G):
      if cm.TimeDomOrEqual(self.f_value[sid], self.f_value[s.id]):
        self.AddToBackSet(parent_sid, sid)
        if len(self.collision_dict[sid]) > 0:
          self.BackPropagation(self.all_visited_s[parent_sid], self.collision_dict[sid])
    return
  
  def Search(self, search_limit=np.inf, time_limit=10):
    print("--------- MOLSSM* Search begin , nr = ", self.num_robots, "---------")
    if self.heu_failed:
      print(" xxxxx MOLSSM* direct terminates because heuristics computation failed...")
      output_res = ( 0, [], 0, -1, self.GetRemainTime(), self.max_col_set )
      return dict(), output_res
    self.time_limit = time_limit
    tstart = time.perf_counter()
    self.all_visited_s[self.s_o.id] = self.s_o
    # f_value is also a matrix, but cost_vector is a vector
    self.f_value[self.s_o.id] = self.s_o.G + self.weight*self.GetHeuristic(self.s_o)
    # np.sum() is reasonable?
    self.open_list.add(np.sum(self.f_value[self.s_o.id]), self.s_o.id)
    self.collision_dict[self.s_o.id] = dict()
    self.AddToFrontier(self.s_o)
    search_success = True
    rd = 0
    while(True):
      tnow = time.perf_counter()
      rd = rd + 1
      if (rd > search_limit) or (tnow - tstart > self.GetRemainTime()):
        print(" Fail! timeout! ")
        search_success = False
        break
      if (self.open_list.size()) == 0:
        print(" Done! openlist is empty! ")
        search_success = True
        break
      pop_node = self.open_list.pop()
      curr_s = self.all_visited_s[pop_node[1]]
      # print(curr_s.id,curr_s.G[0],curr_s.locs)
      # if max(curr_s.G[0])>500:
      #   breakpoint()
      self.RefineGoalFrontier(curr_s)
      if self.FilterState(curr_s, self.f_value[curr_s.id]):
        if curr_s.id in self.frontier_map[self.s_f.locs]:
          self.frontier_map[self.s_f.locs].remove(curr_s.id)
        continue
      if(curr_s.locs == self.s_f.locs):
        # breakpoint()
        continue
      # get neighbors
      ngh_ss, ngh_success = self.GetNgh(curr_s, tnow) # neighboring states
      if not ngh_success:
        search_success = False
        break
      # loop over neighbors
      for idx in range(len(ngh_ss)):
        ngh_s = ngh_ss[idx] 
        self.backtrack_dict[ngh_s.id] = curr_s.id
        h_array = self.GetHeuristic(ngh_s)
        f_array = ngh_s.G + self.weight*h_array
        if self.FilterState(ngh_s, f_array):
          if ngh_s.id in self.frontier_map[self.s_f.locs]:
            self.frontier_map[self.s_f.locs].remove(ngh_s.id)
          continue
        self.AddToBackSet(curr_s.id,ngh_s.id)
        # Compute collision set
        # TODO:Collision in LSS is different from that in other MAPF algorithms
        # Because robots are not synchronized for sure in a state
        # print("cur_s.locs:"+str(curr_s.locs))
        # print("ngh_s.locs:"+str(ngh_s.locs))
        ngh_collision_set = self.CollisionCheck(curr_s,ngh_s)
        # print("ngh_collision_set:"+str(ngh_collision_set))
        if ngh_s.id in self.collision_dict:
          for k in ngh_collision_set: # union with prev collision set.
            self.collision_dict[ngh_s.id][k] = 1
        else:
          self.collision_dict[ngh_s.id] = ngh_collision_set # first time, create a dict
        if len(ngh_collision_set) > 0: 
          self.BackPropagation(curr_s, ngh_collision_set)
          continue # this ngh state is in collision
        # if ngh_s.id == 687:
        #   breakpoint()
        if (not self.Pruning(ngh_s, f_array)): 
          self.AddToFrontier(ngh_s)
          self.f_value[ngh_s.id] = ngh_s.G + self.weight*h_array
          self.open_list.add(np.sum(self.f_value[ngh_s.id]), ngh_s.id)
        else: # dominated
          self.DominanceBackprop(ngh_s)
    if True:
      # output jpath is in reverse order, from goal to start
      all_jpath = self.ReconstructPathAll()
      all_cost_vec = dict()
      for k in all_jpath:
        all_cost_vec[k] = self.all_visited_s[k].G.tolist()
      output_res = ( int(rd), all_cost_vec, int(search_success), float(time.perf_counter()-tstart), float(self.GetRemainTime()), self.max_col_set )
      res = (all_jpath, output_res)
      # res = cm.Non_dominant_sub_set(res)
      print(" MOLSSM* search terminates with ", len(res[0]), " solutions.")
      return res
    else:
      output_res = ( int(rd), dict(), int(search_success), float(time.perf_counter()-tstart), float(self.GetRemainTime()), self.max_col_set )
      return dict(), output_res

def RunMoLSSMstarMAPF(grids, sx, sy, gx, gy, cvecs, cost_grids, cdim, w, eps, search_limit, time_limit):
  """
  sx,sy = starting x,y coordinates of agents.
  gx,gy = goal x,y coordinates of agents.
  cdim = M, cost dimension.
  cvecs = cost vectors of agents, the kth component is a cost vector of length M for agent k.
  cost_grids = a tuple of M matrices, the mth matrix is a scaling matrix for the mth cost dimension.
  cost for agent-i to go through an edge c[m] = cvecs[i][m] * cgrids[m][vy,vx], where vx,vy are the target node of the edge.
  w is the heuristic inflation rate. E.g. w=1.0, no inflation, w>1.0, use inflation. 
  eps is useless. set eps=0.
  search_limit is the maximum rounds of expansion allowed. set it to np.inf if you don't want to use.
  time_limit is the maximum amount of time allowed for search (in seconds), typically numbers are 60, 300, etc.
  """

  print("...RunMoLSSMstarMAPF... ")
  truncated_cvecs = list()
  truncated_cgrids = list()

  # ensure cost dimension.
  for idx in range(len(cvecs)):
    truncated_cvecs.append(cvecs[idx][0:cdim])
  for idx in range(cdim):
    truncated_cgrids.append(cost_grids[idx])

  molssm = MoLSSMstar(grids, sx, sy, gx, gy, truncated_cvecs, truncated_cgrids, w, eps, time_limit, True)
  t_remain = molssm.GetRemainTime() # in constructor defined in MoAstarMAPF, compute policy takes time
  return molssm.Search(search_limit, t_remain)
