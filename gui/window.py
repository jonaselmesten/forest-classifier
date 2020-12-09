import threading
import tkinter as tk
from tkinter import ttk

from PIL import ImageTk, Image, ImageEnhance


class LoadModelWindow(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)

        self.title("Tree classifier")
        self.lift()
        self.eval('tk::PlaceWindow . center')

        self.master_frame = tk.Frame(self)
        self.master_frame.rowconfigure(0, minsize=80, weight=1)
        self.master_frame.columnconfigure([0, 1, 2], minsize=80, weight=1)
        self.y_pad = 10

        self.frame = tk.Frame(master=self.master_frame, relief=tk.RAISED)
        self.frame.grid(row=0, column=1, sticky="")

        self.label = tk.Label(master=self.frame, text="Loading models, please wait.")
        self.label.grid(row=4, column=1, pady=self.y_pad)

        self.progress_bar = ttk.Progressbar(master=self.frame, orient="horizontal", length=150, mode="indeterminate")
        self.progress_bar.start(10)
        self.progress_bar.grid(row=5, column=1, pady=10)

        self.frame.grid()
        self.master_frame.grid()


class ZoomedOutWindow(tk.Toplevel):

    def __init__(self, master, pil_img):
        tk.Toplevel.__init__(self, master=master)
        self.resizable(False, False)
        self.title("Zoomed out image - Click to close")

        width, height = pil_img.size

        self.geometry(str(width) + "x" + str(height))

        tk_img = ImageTk.PhotoImage(pil_img)
        img = tk.Label(self, image=tk_img)
        img.image = tk_img

        btn0 = tk.Button(self, image=tk_img, command=self._close_window)
        btn0.image = tk_img
        btn0.place(x=0, y=0)

    def _close_window(self):
        self.destroy()


class TreeWindow(tk.Tk):

    def __init__(self, class_name, tree_list):
        tk.Tk.__init__(self)

        self.tree_list = tree_list
        self.class_name = class_name
        self.remove_count = 0

        self.img_size = 200

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.state("zoomed")
        self.widget_list = []
        self.lift()

        self.master_frame = tk.Frame(self)
        self.master_frame.grid(sticky=tk.NSEW)
        self.master_frame.columnconfigure(0, weight=1)
        self.master_frame.rowconfigure(0, weight=1)

        self.label = tk.Label(master=self.master_frame, text="Remove trees that are not: "
                                                             + class_name + " | Images left:"
                                                             + str(len(self.tree_list)))
        self.label.config(font=("", 30))
        self.label.grid(row=0, column=0)

        # Create a frame for the canvas and scrollbar(s).
        self.scrollbar_frame = tk.Frame(self.master_frame)
        self.scrollbar_frame.grid(column=0, sticky=tk.NSEW)
        self.scrollbar_frame.columnconfigure(0, weight=1)
        self.scrollbar_frame.rowconfigure(0, weight=1)

        # Add a canvas in that frame.
        self.canvas = tk.Canvas(self.scrollbar_frame)
        self.canvas.grid(sticky=tk.NSEW)
        self.canvas.columnconfigure([0, 1, 2], weight=1)

        # Create a vertical scrollbar linked to the canvas.
        self.scrollbar = tk.Scrollbar(self.scrollbar_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=2, sticky=tk.NSEW)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.buttons_frame = tk.Frame(self.canvas, bd=160)
        self._load_images()

        self.canvas.create_window((0, 0), window=self.buttons_frame, anchor=tk.NW)

        self.buttons_frame.update_idletasks()  # Needed to make bbox info available.

        self.bbox = self.canvas.bbox(tk.ALL)  # Get bounding box of canvas with Buttons.
        self.canvas.configure(scrollregion=self.bbox,
                              width=self.winfo_screenwidth(),
                              height=self.winfo_screenheight() - 100)

    def _zoom_out(self, tree):

        image_path = tree[0]
        tree_data = tree[1]
        x_min, y_min, x_max, y_max = tree_data[0], tree_data[1], tree_data[2], tree_data[3]

        # Open image and mark the tree
        main_pil_image = Image.open(image_path)
        tree_img = main_pil_image.crop((x_min, y_min, x_max, y_max))
        main_pil_image = ImageEnhance.Brightness(main_pil_image).enhance(0.5)
        main_pil_image.paste(tree_img, (x_min, y_min, x_max, y_max))

        width, height = main_pil_image.size

        enlarge = 200
        plus_x = 0
        plus_y = 0

        # Create a zoomed out square image
        # X-corrections
        if x_min - enlarge < 0:
            plus_x = enlarge - x_min
            x_min = 0
        else:
            x_min -= 200

        if x_max + enlarge > width:
            minus_x = (x_max + enlarge) - width
            x_max = width + plus_x
            x_min -= minus_x
        else:
            x_max += enlarge + plus_x

        # Y-corrections
        if y_min - enlarge < 0:
            plus_y = enlarge - y_min
            y_min = 0
        else:
            y_min -= enlarge

        if y_max + enlarge > height:
            minus_y = (y_max + enlarge) - height
            y_max = height + plus_y
            y_min -= minus_y
        else:
            y_max += enlarge + plus_y

        zoom_out_img = main_pil_image.crop((x_min, y_min, x_max, y_max)).resize((500, 500))
        ZoomedOutWindow(master=self, pil_img=zoom_out_img)

    def _load_images(self):

        self.buttons_frame.grid_forget()

        row_index = 1
        column_index = 0
        total_trees = 0
        btn_count = 0
        btn_list = []

        pil_img_cache = dict()

        for row in self.tree_list:

            image_path = row[0]
            tree_row = row[1]
            accuracy = tree_row[5]

            # Skip de-classed trees and trees with high score
            if tree_row[4] != self.class_name:
                continue

            # Add open pil-image to cache for loading
            if image_path not in pil_img_cache:
                pil_img_cache[image_path] = Image.open(image_path)

            min_x, min_y, max_x, max_y = tree_row[0], tree_row[1], tree_row[2], tree_row[3]

            main_pil_image = pil_img_cache[image_path]
            pil_img = main_pil_image.crop((min_x, min_y, max_x, max_y)).resize((self.img_size, self.img_size))

            img = ImageTk.PhotoImage(pil_img)

            img_frame = tk.Frame(self.buttons_frame)
            img_frame.grid(row=row_index,
                           column=column_index,
                           sticky=tk.NSEW,
                           padx=0,
                           pady=10)

            button = tk.Button(img_frame,
                               relief=tk.RIDGE,
                               image=img,
                               width=self.img_size,
                               height=self.img_size,
                               command=lambda index=btn_count: self._remove_class(self.tree_list[index]))

            button.image = img
            button.grid(row=0,
                        column=0,
                        sticky=tk.NSEW)

            text = tk.Label(master=img_frame, text=accuracy)
            text.grid(row=1)

            img_button = tk.Button(img_frame,
                                   text="Zoom out",
                                   command=lambda index=btn_count: self._zoom_out(self.tree_list[index]))
            img_button.grid(row=2)
            btn_list.append(img_button)

            column_index += 1
            btn_count += 1
            self.widget_list.append(img_frame)

            total_trees += 1

            if column_index % 8 == 0:
                column_index = 0
                row_index += 1

            if total_trees == 16:
                break

    def _remove_class(self, tree):

        # ['img_path', [0, 392, 50, 479, 'Birch', 75.61432123184204]]
        tree[1][4] = "Tree"

        self.tree_list.remove(tree)
        self.label.config(text="Remove trees that are not: "
                               + self.class_name + " | Images left:"
                               + str(len(self.tree_list)))

        # Close window when empty
        if len(self.tree_list) == 0:
            self.destroy()
            return

        for w in self.buttons_frame.winfo_children():
            w.grid_forget()
        self._load_images()
