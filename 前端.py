# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import scrolledtext
import time
import ttkbootstrap as ttk
from ttkbootstrap.constants import *


class WeChatUI:
    def __init__(self, master):
        self.master = master
        master.title("微信 (模拟)")
        master.geometry("400x600")  # 设置窗口大小


        # 1. 聊天记录显示区域
        self.chat_log = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=40, height=25)
        self.chat_log.pack(pady=10, padx=10, fill=BOTH, expand=True)
        self.chat_log.config(state=tk.DISABLED)  # 初始状态设为只读

        # 2. 消息输入框
        self.input_frame = ttk.Frame(master)  # 创建一个 Frame 来放置输入框和按钮
        self.input_frame.pack(fill=tk.X, padx=5) # 填满 X 轴，增加一些 padding

        self.message_entry = ttk.Entry(self.input_frame)
        self.message_entry.pack(side=LEFT, fill=BOTH, expand=True, padx=5)  # 占据剩余空间

        # 3. 发送按钮
        self.send_button = ttk.Button(self.input_frame, text="发送", command=self.send_message, bootstyle=PRIMARY)
        self.send_button.pack(side=RIGHT, padx=5)

        # 绑定事件，按回车键发送消息
        self.message_entry.bind("<Return>", self.send_message)


    def send_message(self, event=None):  # event 来自于绑定事件
        message = self.message_entry.get()
        if message:
            self.display_message("我: " + message)  # 显示我发送的消息
            # 在这里添加你自己的逻辑处理消息
            self.process_message(message)
            self.message_entry.delete(0, tk.END)  # 清空输入框

    def display_message(self, message):
        self.chat_log.config(state=tk.NORMAL)  # 设置为可编辑状态
        self.chat_log.insert(tk.END, message + "\n")
        self.chat_log.config(state=tk.DISABLED)  # 恢复为只读状态
        self.chat_log.see(tk.END)  # 滚动到底部

    def process_message(self, message):
        # 模拟简单的回应
        if "你好" in message:
            response = "你好！有什么可以帮助你的？"
        elif "时间" in message:
            response = "现在时间是: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        else:
            response = "我不太明白你的意思。"
        self.master.after(1000, lambda: self.display_message("微信助手: " + response)) # 模拟延迟回复 (1秒)


root = ttk.Window(themename="darkly")  #  "darkly", "flatly", "journal", "litera", "lumen", "minty", "pulse", "sandstone", "simplex", "sketchy", "solar", "spacelab", "united", "yeti"
ui = WeChatUI(root)
root.mainloop()
