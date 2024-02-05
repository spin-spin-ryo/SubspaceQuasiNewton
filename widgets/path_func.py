import os
import numpy as np
from utils.save_func import get_params_from_path,load_config
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
SLASH = os.path.join("a","b")[1:-1]
import jax.numpy as jnp

def show_result_with_option(result_pathes,options):
  fvalues = []
  time_values = []
  labeledflag = False
  labeled = {}
  start = 0
  end = -1
  mode = "best"
  xscale = ""
  yscale = ""
  full_line = 100
  LABELFONTSIZE = 18
  TICKLABELSIZE = 18
  plt.rcParams["font.family"] = 'Times New Roman'
  plt.rcParams["mathtext.fontset"] = 'stix'
  plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
  plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
  plt.gca().ticklabel_format(style="sci",scilimits=(0,0))
    
  for result_path in result_pathes:
    params = result_path.split(SLASH)
    labeled[result_path] = SLASH.join(params[-2:])
        
    #option関連
  for k,v in options.items():
    if k == "start":
      start = v
    if k == "end":
      end = v
    if k == "xscale":
      xscale = v
    if k == "yscale":
      yscale = v
    if k == "full_line":
      full_line = v
    if k == "mode":
      mode = v
    if k== "label_fontsize":
      LABELFONTSIZE = v
    if k == "tick_fontsize":
      TICKLABELSIZE = v
    if k == "label":
      labeledflag = v
      for result_path in result_pathes:
        if v:
          solver_name = result_path.split(SLASH)[-2]
          labeled[result_path] = solver_name
                
  for result_path in result_pathes:
    print(result_path)
    if mode == "best":
      fvalues.append(jnp.load(os.path.join(result_path,"func_values.npy")))
      time_values.append(jnp.load(os.path.join(result_path,"time.npy")))
      
  if "time" in xscale:
    for index,(p,v,t) in enumerate(zip(result_pathes,fvalues,time_values)):
      nonzeroindex = t>0
      v = v[nonzeroindex]
      t = t[nonzeroindex]
      if end != -1:
        index = t < end
      else:
        index = np.ones(len(t),dtype=bool)    
      # start = 0　想定
      if "proposed" in p:
        plt.plot(t[index][::full_line],v[index][::full_line],label = labeled[p])
      else:
        plt.plot(t[index][::full_line],v[index][::full_line],label = labeled[p],linestyle = "dotted")
    plt.xlabel("Time[s]",fontsize = LABELFONTSIZE)
  else:
    for index,(p,v,t) in enumerate(zip(result_pathes,fvalues,time_values)):
      nonzeroindex = t > 0
      v = v[nonzeroindex]
      if "proposed" in p:
        plt.plot(np.arange(len(v))[start:end][::full_line],v[start:end][::full_line],label = labeled[p])
      else:
        plt.plot(np.arange(len(v))[start:end][::full_line],v[start:end][::full_line],label = labeled[p],linestyle = "dotted")
    plt.xlabel("Iterations",fontsize = LABELFONTSIZE)
  
  plt.rc('font', size=TICKLABELSIZE)
  if not labeledflag:
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=1,borderaxespad=0,fontsize = LABELFONTSIZE)
  else:
    plt.legend()
  plt.ylabel(r'$f(x)$',fontsize = LABELFONTSIZE)
  if "log" in xscale:
    plt.xscale("log")
  if yscale == "log":
    plt.yscale("log")
  
  plt.tight_layout()
  plt.show()
  


def get_best_result_path(init_dir,prop_dict):
  # init_dir以下でprop_dictで指定されている要素の中から最適解のpathを見つけてくる.
  # init_dirはsolver_nameまでのdirで
  dir_list = os.listdir(init_dir)
  min_val = None
  min_val_dir = None
  for dir_name in dir_list:
    now_prop_dict = get_params_from_path(dir_name)
    ok_flag = True
    # check
    for k,v in prop_dict.items():
      if v != "" and v != now_prop_dict[k]:
        ok_flag = False
    if ok_flag:
      now_val = get_min_val_from_result(os.path.join(init_dir,dir_name,"result.json"))
      if np.isnan(now_val):
        continue
      if min_val is None:
        min_val = now_val
        min_val_dir = dir_name
      else:
        if now_val < min_val:
          min_val = now_val
          min_val_dir = dir_name
  return min_val_dir,min_val

def get_min_val_from_result(file_name):
  min_val = 10000000
  results = load_config(file_name)
  results = results["result"]
  for result_dict in results:
    save_values_list = result_dict["save_values"]
    for save_values in save_values_list:
      for k,v in save_values.items():
        if k == "min_value":
          min_val = min(min_val,v)
  return min_val