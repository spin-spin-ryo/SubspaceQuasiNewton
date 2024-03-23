import os
import numpy as np
from utils.save_func import get_params_from_path,load_config
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
SLASH = os.path.join("a","b")[1:-1]
import jax.numpy as jnp
from environments import *

FORMAL_LABEL = {
  GRADIENT_DESCENT:GRADIENT_DESCENT,
  SUBSPACE_GRADIENT_DESCENT:SUBSPACE_GRADIENT_DESCENT,
  ACCELERATED_GRADIENT_DESCENT:ACCELERATED_GRADIENT_DESCENT,
  NEWTON:NEWTON,
  SUBSPACE_NEWTON:SUBSPACE_NEWTON,
  LIMITED_MEMORY_NEWTON:LIMITED_MEMORY_NEWTON,
  LIMITED_MEMORY_BFGS:LIMITED_MEMORY_BFGS,
  BFGS_QUASI_NEWTON:BFGS_QUASI_NEWTON,
  RANDOM_BFGS:RANDOM_BFGS,
  SUBSPACE_REGULARIZED_NEWTON:SUBSPACE_REGULARIZED_NEWTON,
  PROXIMAL_GRADIENT_DESCENT:PROXIMAL_GRADIENT_DESCENT,
  ACCELERATED_PROXIMAL_GRADIENT_DESCENT:ACCELERATED_GRADIENT_DESCENT,
  MARUMO_AGD:MARUMO_AGD,
  SUBSPACE_QUASI_NEWTON:SUBSPACE_QUASI_NEWTON
}

MARKERS = {
  GRADIENT_DESCENT:"x",
  SUBSPACE_GRADIENT_DESCENT:"s",
  ACCELERATED_GRADIENT_DESCENT:"o",
  NEWTON:"v",
  SUBSPACE_NEWTON:"^",
  LIMITED_MEMORY_NEWTON:"<",
  LIMITED_MEMORY_BFGS:">",
  BFGS_QUASI_NEWTON:"D",
  RANDOM_BFGS:"8",
  SUBSPACE_REGULARIZED_NEWTON:"+",
  PROXIMAL_GRADIENT_DESCENT:".",
  ACCELERATED_PROXIMAL_GRADIENT_DESCENT:"o",
  MARUMO_AGD:"o",
  SUBSPACE_QUASI_NEWTON:""
}

MARKERS_LIST = ["",
                "o",
                "x",
                "s",
                "v",
                "*",
                "+",
                "^"]

def show_result_with_option(result_pathes,options):
  fvalues = []
  time_values = []
  labeledflag = False
  labeled = {}
  start = 0
  end = -1
  mode = "function_value"
  xscale = ""
  yscale = ""
  full_line = 100
  LABELFONTSIZE = 18
  TICKLABELSIZE = 18
  LEGENDFONTSIZE = 18
  are_all_proposed = True
  plt.figure()
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
    if k == "legend_fontsize":
      LEGENDFONTSIZE = v
    if k == "label":
      labeledflag = v
      for result_path in result_pathes:
        solver_name = result_path.split(SLASH)[-2]
        are_all_proposed = ("Proposed" == solver_name)

      for result_path in result_pathes:
        if v:
          solver_name = result_path.split(SLASH)[-2]
          param_dir = result_path.split(SLASH)[-1]
          param_dict = get_params_from_path(param_dir)
          if solver_name == "Proposed":
            matrix_size = param_dict["matrix_size"]
            reduced_dim = param_dict["reduced_dim"]
            if are_all_proposed:
              labeled[result_path] = r"$(d = {}, m = {})$".format(reduced_dim,matrix_size)
            else:
              labeled[result_path] = FORMAL_LABEL[solver_name]+r"$(d = {}, m = {})$".format(reduced_dim,matrix_size)
          else:
            labeled[result_path] = FORMAL_LABEL[solver_name]
                
  for result_path in result_pathes:
    print(result_path)
    if mode == "function_value":
      fvalues.append(jnp.load(os.path.join(result_path,"func_values.npy")))
      time_values.append(jnp.load(os.path.join(result_path,"time.npy")))
    elif mode == "grad_norm":
      fvalues.append(jnp.load(os.path.join(result_path,"grad_norm.npy")))
      time_values.append(jnp.load(os.path.join(result_path,"time.npy")))
      
  for index,(p,v,t) in enumerate(zip(result_pathes,fvalues,time_values)):
    nonzeroindex = t>0
    nonzeroindex = np.array(nonzeroindex)
    nonzeroindex[0] = True
    v = v[nonzeroindex]
    solver_name = p.split(SLASH)[-2]
    if "time" in xscale:
      t = t[nonzeroindex]
      t+=1
      x_list = t
      y_list = v   
      plt.xlabel("Time[s]",fontsize = LABELFONTSIZE)
    else:
      x_list = np.arange(1,len(v)+1)
      y_list = v
      plt.xlabel("Iterations",fontsize = LABELFONTSIZE)
    
    if full_line*10 < len(v):
      x_list = x_list[::full_line]
      y_list = y_list[::full_line]
    
    if not are_all_proposed:
      plt.plot(x_list,y_list,marker = MARKERS[solver_name],label = labeled[p])
    else:
      plt.plot(x_list,y_list,marker = MARKERS_LIST[index],label = labeled[p])

  if end != -1:
    plt.xlim(left = start-end*0.1,right = end*1.1)
  plt.rc('font', size=TICKLABELSIZE)
  if not labeledflag:
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=1,borderaxespad=0,fontsize = LEGENDFONTSIZE)
  else:
    plt.legend(fontsize = LEGENDFONTSIZE)
  if mode == "function_value":
    plt.ylabel(r'$f(x)$',fontsize = LABELFONTSIZE)
  elif mode == "grad_norm":
    plt.ylabel(r'$\|\nabla f(x)\|$',fontsize = LABELFONTSIZE)
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