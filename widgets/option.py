import tkinter as tk

def generate_entry(parent,text):
    frame = tk.Frame(parent)
    tk.Label(frame,text=text).pack(side = "left")
    entry = tk.Entry(frame)
    entry.pack(side = "left")
    frame.pack()
    return entry


def open_option_window(root,option_window,option_entries):
    if option_window is None:
        option_window = tk.Toplevel(root)
    else:
        option_window.lift()
        return
    entry_start = generate_entry(option_window,"start")
    entry_end = generate_entry(option_window,"end")
    entry_xscale = generate_entry(option_window,"xscale")
    entry_yscale = generate_entry(option_window,"yscale")
    entry_interval = generate_entry(option_window,"full_line")
    entry_mode = generate_entry(option_window,"mode")
    entry_label = generate_entry(option_window,"label")
    entry_labelfontsize = generate_entry(option_window,"label_fontsize")
    entry_tickfontsize = generate_entry(option_window,"tick_fontsize")
    entry_legendfontsize = generate_entry(option_window,"legend_fontsize")
    entry_start.insert(0,"0")
    entry_end.insert(0,"-1")
    entry_interval.insert(0,"100")
    entry_mode.insert(0,"function_value")
    entry_labelfontsize.insert(0,"18")
    entry_tickfontsize.insert(0,"18")
    entry_legendfontsize.insert(0,"18")
    entry_label.insert(0,"True")

    

    option_entries[("start",int)] = entry_start
    option_entries[("end",int)] = entry_end
    option_entries[("xscale",str)] = entry_xscale
    option_entries[("yscale",str)] = entry_yscale
    option_entries[("full_line",int)] = entry_interval
    option_entries[("mode",str)] = entry_mode
    option_entries[("label",bool)] = entry_label
    option_entries[("label_fontsize",int)] = entry_labelfontsize
    option_entries[("tick_fontsize",int)] = entry_tickfontsize
    option_entries[("legend_fontsize",int)] = entry_legendfontsize
    return option_window,option_entries
