import tkinter as tk
import tkinter.ttk as ttk
import os
from environments import RESULTPATH
from utils.save_func import get_params_from_path,get_path_form_params
from problems.generate_problem import objective_properties_key,constraints_properties_key

class Labeled_cmbbox:
  def __init__(self,parent,text,values) -> None:
    self.parent = parent
    self.frame = tk.Frame(self.parent)
    self.label = tk.Label(self.frame,text=text)
    self.cmbbox = ttk.Combobox(self.frame,values=values,state="readonly")
    pass

  def pack(self):
    self.frame.pack()
    self.label.grid(row = 0,column=0)
    self.cmbbox.grid(row = 0,column=1)
  
  def get_value_of_param(self):
    return self.cmbbox.get()

class select_objective_box:
  def __init__(self,parent) -> None:
    self.parent = parent
    self.frame = tk.Frame(self.parent)
    names = self.get_objective_names()
    self.cmbbox = ttk.Combobox(self.frame,values=names,state="readonly")
    self.selected_objective = None
    self.params_box = {}
    self.cmbbox.bind(" <<ComboboxSelected>> ",self.get_properties_of_objective)
  
  def get_objective_names(self):
    dirs = os.listdir(RESULTPATH)
    return dirs
  
  def get_properties_of_objective(self):
    self.selected_objective = self.cmbbox.get()
    dirs = os.listdir(RESULTPATH,self.selected_objective)
    values_dict = {}
    for param in objective_properties_key[self.selected_objective]:
      values_dict[param] = set()

    for dir_name in dirs:
      params_dict = get_params_from_path(dir_name)
      for k,v in params_dict.items():
        values_dict[k].add(v)
    
    for param,values in values_dict.items():
      cmbbox = Labeled_cmbbox(self.frame,text=param,values=values)
      cmbbox.pack()
      self.params_box[param] = cmbbox
  
  def get_values_of_params(self):
    params_dict = {}
    for param,cmbbox in self.params_box.items():
      params_dict[param] = cmbbox.get()
    
    selected_dir_name = get_path_form_params(params_dict)
    return selected_dir_name
  
  def pack(self):
    self.cmbbox.pack()
    self.frame.pack()

class select_constraints_box:
  def __init__(self,parent,function_box) -> None:
    self.parent = parent
    self.frame = tk.Frame(self.parent)
    self.function_box = function_box
    names = self.get_constraints_names()
    self.cmbbox = ttk.Combobox(self.frame,values=names,state="readonly")
    self.selected_constraints = None
    self.params_box = {}
    self.cmbbox.bind(" <<ComboboxSelected>> ",self.get_properties_of_constraints)

  def get_constraints_names(self):
    dirs = os.listdir(RESULTPATH,self.function_box.selected_objective,self.function_box.get_values_of_params())
    return dirs
  
  def get_properties_of_constraints(self):
    self.selected_constraints = self.cmbbox.get()
    dirs = os.listdir(RESULTPATH,self.function_box.selected_objective,self.function_box.get_values_of_params())
    values_dict = {}
    for param in objective_properties_key[self.selected_constraints]:
      values_dict[param] = set()

    for dir_name in dirs:
      params_dict = get_params_from_path(dir_name)
      for k,v in params_dict.items():
        values_dict[k].add(v)
    
    for param,values in values_dict.items():
      cmbbox = Labeled_cmbbox(self.frame,text=param,values=values)
      cmbbox.pack()
      self.params_box[param] = cmbbox
  
  def get_values_of_params(self):
    params_dict = {}
    for param,cmbbox in self.params_box.items():
      params_dict[param] = cmbbox.get()
    
    selected_dir_name = get_path_form_params(params_dict)
    return selected_dir_name




class main_window:
  def __init__(self) -> None:
    self.root = tk.Tk()
    self.root.geometry("400x800")
    self.select_objective_frame = select_objective_box(self.root)
    self.select_objective_frame.pack()
    self.select_constraints_frame = None
    self.set_constraints_button = tk.Button(self.root,text="set constraints")
  
  def set_constraints(self):
    self.select_constraints_frame = select_constraints_box(self.root,self.select_objective_frame)
    # self.select_constraints_frame.pack()
    



  
  def mainloop(self):
    self.root.mainloop()