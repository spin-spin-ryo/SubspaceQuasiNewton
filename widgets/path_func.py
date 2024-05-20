import os
import numpy as np
from utils.save_func import get_params_from_path,load_config
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
SLASH = os.path.join("a","b")[1:-1]
import jax.numpy as jnp
from environments import *
import re
from scipy import interpolate

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
  y_bottom = None
  y_top = None
  mode = "function_value"
  style = "normal"
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
    if k == "style":
      style = v
    if k == "y_bottom":
      y_bottom = v
    if k == "y_top":
      y_top = v
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
        are_all_proposed = are_all_proposed and (SUBSPACE_QUASI_NEWTON == solver_name)

      for result_path in result_pathes:
        if v:
          solver_name = result_path.split(SLASH)[-2]
          param_dir = result_path.split(SLASH)[-1]
          param_dict = get_params_from_path(param_dir)
          if solver_name == SUBSPACE_QUASI_NEWTON:
            matrix_size = param_dict["matrix_size"]
            reduced_dim = param_dict["reduced_dim"]
            if are_all_proposed:
              labeled[result_path] = r"$(d = {}, m = {})$".format(reduced_dim,matrix_size)
            else:
              labeled[result_path] = FORMAL_LABEL[solver_name]+r" $(d = {}, m = {})$".format(reduced_dim,matrix_size)
          elif solver_name == SUBSPACE_QUASI_NEWTON:
            reduced_dim = param_dict["reduced_dim"]
            labeled[result_path] = FORMAL_LABEL[solver_name]+r" $(d = {})$".format(reduced_dim)
          elif solver_name == SUBSPACE_GRADIENT_DESCENT:
            reduced_dim = param_dict["reduced_dim"]
            labeled[result_path] = FORMAL_LABEL[solver_name]+r" $(d = {})$".format(reduced_dim)
          elif solver_name == LIMITED_MEMORY_NEWTON:
            matrix_size = param_dict["reduced_dim"]
            labeled[result_path] = FORMAL_LABEL[solver_name]+r" $(m = {})$".format(matrix_size)
            
          else:
            labeled[result_path] = FORMAL_LABEL[solver_name]
                
  # for result_path in result_pathes:
  #   print(result_path)
  #   if mode == "function_value":
  #     fvalues.append(jnp.load(os.path.join(result_path,"func_values.npy")))
  #     time_values.append(jnp.load(os.path.join(result_path,"time.npy")))
  #   elif mode == "grad_norm":
  #     fvalues.append(jnp.load(os.path.join(result_path,"grad_norm.npy")))
  #     time_values.append(jnp.load(os.path.join(result_path,"time.npy")))
      
  for index,p in enumerate(result_pathes):
    if style == "normal":
      t = jnp.load(os.path.join(p,"time.npy"))
      if mode == "function_value":
        v = jnp.load(os.path.join(result_path,"func_values.npy"))
      elif mode == "grad_norm":
        v = jnp.load(os.path.join(result_path,"grad_norm.npy"))
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
      else:
        x_list = np.arange(1,len(v)+1)
        y_list = v
        
      if full_line*10 < len(v):
        x_list = x_list[::full_line]
        y_list = y_list[::full_line]
      
      if not are_all_proposed:
        plt.plot(x_list,y_list,marker = MARKERS[solver_name],label = labeled[p])
      else:
        plt.plot(x_list,y_list,marker = MARKERS_LIST[index],label = labeled[p])
    elif (style == "error" or style == "average") and os.path.exists(os.path.join(p,"others")):
      data_dir = os.path.join(p,"others")
      results = get_results_from_dir(data_dir)
      results_numpy = transform_dict_to_numpy(results)
      if mode == "function_value":
        y_list = results_numpy["func_values"]
      elif mode == "grad_norm":
        y_list = results_numpy["grad_norm"]

      if "time" in xscale:
        x_list = results_numpy["time"]
      else:
        x_list = np.arange(1,len(y_list[0])+1).reshape(1,-1)
      std = 1
      if style == "error":
        std = 1
      elif style == "average":
        std = 0
      shadow_plot(y_list,x_list,marker = "",label = labeled[p],std = std)
    elif style == "all" and os.path.exists(os.path.join(p,"others")):
      data_dir = os.path.join(p,"others")
      results = get_results_from_dir(data_dir)
      results_numpy = transform_dict_to_numpy(results)
      if mode == "function_value":
        y_list = results_numpy["func_values"]
      elif mode == "grad_norm":
        y_list = results_numpy["grad_norm"]

      if "time" in xscale:
        x_list = results_numpy["time"]
      else:
        x_list = np.arange(1,len(y_list[0])+1).reshape(1,-1)
      for index in range(len(y_list)):
        if len(x_list) == 1:
          plt.plot(x_list[0],y_list[index])
        else:
          plt.plot(x_list[index],y_list[index])
    else:
      t = jnp.load(os.path.join(p,"time.npy"))
      if mode == "function_value":
        v = jnp.load(os.path.join(p,"func_values.npy"))
      elif mode == "grad_norm":
        v = jnp.load(os.path.join(p,"grad_norm.npy"))
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
      else:
        x_list = np.arange(1,len(v)+1)
        y_list = v
        
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
  
  if labeledflag:
    plt.legend(fontsize = LEGENDFONTSIZE)
  
  if "time" in xscale:
    plt.xlabel("Time[s]",fontsize = LABELFONTSIZE)
  else:
    plt.xlabel("Iterations",fontsize = LABELFONTSIZE)
      
  if mode == "function_value":
    plt.ylabel(r'$f(x)$',fontsize = LABELFONTSIZE)
  elif mode == "grad_norm":
    plt.ylabel(r'$\|\nabla f(x)\|$',fontsize = LABELFONTSIZE)
  if "log" in xscale:
    plt.xscale("log")
  if yscale == "log":
    plt.yscale("log")
  
  if len(y_bottom)!= 0 :
    plt.ylim(bottom = float(y_bottom))
  if len(y_top) != 0:
    plt.ylim(top = float(y_top))
  
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

def get_results_from_dir(data_dir):
  results = {}
  file_names = os.listdir(data_dir)
  for file_name in file_names:
    last_extension = file_name.split(".")[-1]
    if last_extension == "txt":
      continue
    file_name_without_extension = file_name.split(".")[0]
    UID = re.findall(r'\d+', file_name_without_extension)[0]
    file_name_without_extension = file_name_without_extension.replace(UID,"")
    UID = int(UID)
    if UID in results.keys():
      pass
    else:
      results[UID] = {}
    results[UID][file_name_without_extension] = np.load(os.path.join(data_dir,file_name))
  removeUIDS = []
  for UID in results.keys():
    if np.isnan(np.max(results[UID]["func_values"])):
      removeUIDS.append(UID)
  
  if len(removeUIDS) != 0:
    print("remove data")
    for UID in removeUIDS:
      results.pop(UID)
  return results

def transform_dict_to_numpy(results):
  results_numpy = {}
  data_size = len(results)
  for index,UID in enumerate(results.keys()):
    data_keys = list(results[UID].keys())
    for data_name in data_keys:
      if index == 0:
        array_length = len(results[UID][data_name])
        results_numpy[data_name] = np.empty((data_size,array_length))
      results_numpy[data_name][index] = results[UID][data_name]
  return results_numpy
  
def shadow_plot(y_data,x_data,alpha=0.5,marker = "",label = "",std = 1):
  _y_data = y_data.copy()
  x = None
  y_ave = None
  y_std = None
  use_data_index = 0
  if len(x_data) != 1:
    min_last_time = x_data[use_data_index][-1]
    for index in range(len(x_data)):
      if min_last_time > x_data[index][-1]:
        min_last_time = x_data[index][-1]
        use_data_index = index
  x = x_data[use_data_index]
  if len(x_data) != 1:  
    for index in range(len(y_data)):
      f = interpolate.interp1d(x_data[index],y_data[index])
      _y_data[index] = f(x)
  
  y_ave = np.mean(_y_data,axis = 0)
  y_std = np.std(_y_data,axis = 0)
  plt.plot(x,y_ave,marker = marker,label = label)
  plt.fill_between(x,y_ave + std*y_std,y_ave - std*y_std,alpha=alpha)