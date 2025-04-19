# -*- coding: utf-8 -*-
import ttkbootstrap
from ttkbootstrap.constants import *

root = ttkbootstrap.Window(themename="darkly")  # 测试是否能创建窗口
root.title("TTKBootstrap Test")
ttk.Button(root, text="Click Me", bootstyle="primary").pack()
root.mainloop()