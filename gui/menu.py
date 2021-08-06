import threading
import tkinter as tk
from tkinter.ttk import Progressbar


class MainWin(tk.Tk, threading.Thread):

    def __init__(self, add_images, run_prediction, annotate_prediction):
        tk.Tk.__init__(self)
        threading.Thread.__init__(self)
        self.start()

        self.title("Tree classifier")
        self.lift()
        self.eval('tk::PlaceWindow . center')

        master_frame = tk.Frame(self)
        master_frame.rowconfigure(0, minsize=150, weight=1)
        master_frame.columnconfigure([0, 1, 2], minsize=200, weight=1)
        y_pad = 15
        width = 15
        height = 3

        self.frame = tk.Frame(
            master=master_frame,
            relief=tk.RAISED,
        )

        self.frame.grid(row=0, column=1, sticky="")

        add_img = tk.Button(master=self.frame,
                            width=width,
                            height=height,
                            text="Add images",
                            command=add_images)
        add_img.grid(row=0, column=1, pady=y_pad)

        run_pred = tk.Button(master=self.frame,
                             width=width,
                             height=height,
                             text="Run predictions",
                             command=run_prediction)
        run_pred.grid(row=1, column=1, pady=y_pad)

        ann_pred = tk.Button(master=self.frame,
                             width=width,
                             height=height,
                             text="Annotate prediction",
                             command=annotate_prediction)
        ann_pred.grid(row=2, column=1, pady=y_pad)

        self.label = tk.Label(master=self.frame, text="")
        self.label.grid(row=4, column=1, pady=y_pad)

        self.progress_bar = Progressbar(master=self.frame,
                                        orient="horizontal",
                                        length=150, mode="determinate")

        self.frame.grid()
        master_frame.grid()

    def update_text(self, update):
        self.update_idletasks()
        self.label["text"] = update
        self.update()

    def start_progress_bar(self):
        self.progress_bar.grid(row=5, column=1, pady=10)
        self.update()

    def inc_progress_bar(self, value):
        self.progress_bar["value"] += value
        self.update()
