import tkinter as tk
import tkinter.ttk as ttk
import os
from environments import RESULTPATH
from utils.save_func import get_params_from_path,get_path_form_params
from problems.generate_problem import objective_properties_key,constraints_properties_key
from widgets.option import open_option_window
from widgets.path_func import get_best_result_path,show_result_with_option

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
  
  def destroy(self):
    self.label.destroy()
    self.cmbbox.destroy()
    self.frame.destroy()

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
  
  def get_selected_objective_name(self):
    return self.selected_objective
  
  def get_properties_of_objective(self,event):
    self.selected_objective = self.cmbbox.get()
    dirs = os.listdir(os.path.join(RESULTPATH,self.selected_objective))
    values_dict = {}
    for param in objective_properties_key[self.selected_objective]:
      values_dict[param] = set()

    for dir_name in dirs:
      params_dict = get_params_from_path(dir_name)
      for k,v in params_dict.items():
        values_dict[k].add(v)
    
    for param in self.params_box.keys():
      self.params_box[param].destroy()
    
    self.params_box = {}
    for param,values in values_dict.items():
      cmbbox = Labeled_cmbbox(self.frame,text=param,values=list(values))
      cmbbox.pack()
      self.params_box[param] = cmbbox
  
  def get_values_of_params(self):
    params_dict = {}
    for param,cmbbox in self.params_box.items():
      params_dict[param] = cmbbox.get_value_of_param()
    
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
    dirs = os.listdir(os.path.join(RESULTPATH,self.function_box.selected_objective,self.function_box.get_values_of_params()))
    return dirs
  
  def get_selected_constraint_name(self):
    return self.selected_constraints
  
  def get_properties_of_constraints(self,event):
    self.selected_constraints = self.cmbbox.get()
    dirs = os.listdir(os.path.join(RESULTPATH,self.function_box.selected_objective,self.function_box.get_values_of_params(),self.selected_constraints))
    values_dict = {}
    for param in constraints_properties_key[self.selected_constraints]:
      values_dict[param] = set()

    for dir_name in dirs:
      params_dict = get_params_from_path(dir_name)
      for k,v in params_dict.items():
        values_dict[k].add(str(v))
    
    for param in self.params_box.keys():
      self.params_box[param].destroy()
    self.params_box = {}
    
    for param,values in values_dict.items():
      cmbbox = Labeled_cmbbox(self.frame,text=param,values=list(values))
      cmbbox.pack()
      self.params_box[param] = cmbbox
  
  def get_values_of_params(self):
    params_dict = {}
    for param,cmbbox in self.params_box.items():
      params_dict[param] = cmbbox.get_value_of_param()
    
    selected_dir_name = get_path_form_params(params_dict)
    return selected_dir_name

  def pack(self):
    self.cmbbox.pack()
    self.frame.pack()

class solver_notebook:
  def __init__(self,parent,problem_path) -> None:
    self.parent = parent
    self.problem_path = problem_path
    self.notebook = ttk.Notebook(self.parent)
    self.pages = []
    pass

  def add_page(self):
    count = len(self.pages)+1
    frame = tk.Frame(self.notebook)
    page = page_content(frame,self.problem_path)
    self.pages.append(page)
    self.notebook.add(frame,text = str(count))
  
  def remove_page(self):
    self.notebook.forget(len(self.pages)-1)
    self.pages.pop()
  
  def pack(self):
    self.notebook.pack()

class page_content:
  def __init__(self,parent,problem_path) -> None:
    self.parent = parent
    self.problem_path = problem_path
    self.solvers = os.listdir(problem_path)
    self.frame = tk.Frame(self.parent)
    self.cmbbox = ttk.Combobox(self.frame,values=self.solvers,state="readonly")
    self.cmbbox.bind(" <<ComboboxSelected>> ",self.get_params_of_algorithm)
    self.frame.pack()
    self.cmbbox.pack()
    self.params_box = {}
    pass

  def get_solver_name(self):
    return self.cmbbox.get()

  def get_params_of_algorithm(self,event):
    solver_name = self.cmbbox.get()
    solver_path = os.path.join(self.problem_path,solver_name)
    dirs = os.listdir(solver_path)
    values_dict = {}
    for dir_name in dirs:
      params_dict = get_params_from_path(dir_name)
      for k,v in params_dict.items():
        try:
          values_dict[k].add(str(v))
        except KeyError:
          values_dict[k] = set()
          values_dict[k].add(str(v))
    
    for param in self.params_box.keys():
      self.params_box[param].destroy()
    self.params_box = {}
    for param,values in values_dict.items():
      cmbbox = Labeled_cmbbox(self.frame,text=param,values=list(values))
      cmbbox.pack()
      self.params_box[param] = cmbbox
  
  
  def get_result_path(self):
    solver_name = self.cmbbox.get()
    solver_dir = os.path.join(self.problem_path,solver_name)
    params_dict = {}
    for param,cmbbox in self.params_box.items():
      params_dict[param] = cmbbox.get_value_of_param()
    
    selected_dir_name,min_val = get_best_result_path(solver_dir,params_dict)
    print(solver_name,min_val)
    return os.path.join(self.problem_path,solver_name,selected_dir_name)

class main_window:
  def __init__(self) -> None:
    self.root = tk.Tk()
    self.root.geometry("400x800")
    self.select_objective_frame = select_objective_box(self.root)
    self.select_objective_frame.pack()
    self.select_constraints_frame = None
    self.set_constraints_button = tk.Button(self.root,text="set constraints",command = self.set_constraints)
    self.set_solver_button = tk.Button(self.root,text = "set solver",command=self.set_solver)
    self.set_constraints_button.pack()
    self.notebook = None
    self.page_add_button = tk.Button(self.root,text="add")
    self.page_remove_button = tk.Button(self.root,text="remove")
    self.option_button = tk.Button(self.root,text="option",command=self.open_option)
    self.option_window = None
    self.option_entries = {}
    self.show_result_button = tk.Button(self.root,text="Show",command=self.show_result)
    
  
  def set_constraints(self):
    self.select_constraints_frame = select_constraints_box(self.root,self.select_objective_frame)
    self.select_constraints_frame.pack()
    self.set_solver_button.pack()
  
  def show_result(self):
    result_pathes = []
    for page in self.notebook.pages:
      result_path = page.get_result_path()
      result_pathes.append(result_path)
    
    options = {}
    for k,v in self.option_entries.items():
      key,t = k
      options[key] = t(v.get())
    
    show_result_with_option(result_pathes=result_pathes,options=options)
  
  def set_solver(self):
    objective_name = self.select_objective_frame.get_selected_objective_name()
    objective_properties = self.select_objective_frame.get_values_of_params()
    constriants_name = self.select_constraints_frame.get_selected_constraint_name()
    constraints_properties = self.select_constraints_frame.get_values_of_params()
    problem_path = os.path.join(RESULTPATH,objective_name,objective_properties,constriants_name,constraints_properties)
    self.notebook = solver_notebook(self.root,problem_path=problem_path)
    self.notebook.pack()
    self.page_add_button.config(command=self.notebook.add_page)
    self.page_add_button.update()
    self.page_remove_button.config(command=self.notebook.remove_page)
    self.page_remove_button.update()
    self.page_add_button.pack()
    self.page_remove_button.pack()
    self.option_button.pack()
    self.show_result_button.pack()

  def open_option(self):
    self.option_window,self.option_entries = open_option_window(self.root,self.option_window,self.option_entries)
  
  def mainloop(self):
    self.root.mainloop()