from tkinter import messagebox, filedialog
import tkinter as tk
from tkinter import ttk
from tkinter import Toplevel

import customtkinter
import math
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from osgeo import gdal
from PyQt5.QtWidgets import QApplication,QMessageBox, QScrollArea,QWidget, QPushButton,QProgressDialog, QLabel, QVBoxLayout, QFileDialog,QTextEdit,QHBoxLayout
from PyQt5.QtGui import QPixmap,QImage
from PyQt5 import QtCore
import cv2
import numpy as np
from PyQt5.QtCore import Qt
import simplekml
import rasterio
from pyproj import Transformer
from PIL import Image, ImageFile
import os
import xlsxwriter
from PIL import Image, ImageTk
import numpy as np
from matplotlib.colors import to_hex
import modal
from tkinter.ttk import Progressbar
import time

Image.MAX_IMAGE_PIXELS = None
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

stop_thread = False
def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

def add_excel(spaces, rows, counts, filename):
    

    # Create a new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook(f'{filename}.xlsx')
    worksheet = workbook.add_worksheet()

    # Some sample data for the pie chart.
    data = [
        ['Section', 'Spacelengh','Rowlength','SpaceCount','Percentage'],
    ]

    total_space=0
    total_row=0
    total_count=0


    for i in range(len(spaces)):
       
        list1 = []
        list1.append('Section '+ str(i+1))
        list1.append(spaces[i])
        list1.append(rows[i])
        list1.append(counts[i])
        list1.append(int(spaces[i]*100/rows[i]))
        data.append(list1)
        total_row += rows[i]
        total_space += spaces[i]
        total_count += counts[i]
    list1 = ['Total',total_space,total_row,total_count,int(total_space*100/total_row)]
    data.append(list1)

    for row_num, row_data in enumerate(data):
        for col_num, cell_data in enumerate(row_data):
            worksheet.write(row_num, col_num, cell_data)

    # Create a new chart object.
    piechart = workbook.add_chart({'type': 'pie'})

    # Configure the series. Note the use of `(sheetname, first_row, first_col, last_row, last_col)` as arguments.
    piechart.add_series({
        'name':       'Section',
        'categories': ['Sheet1', 1, 0, len(spaces), 0],
        'values':     ['Sheet1', 1, 1, len(spaces), 1],
        'data_labels': {
        'percentage': True,
        'legend_key': True,  # Optional: Include the legend keys next to the percentage
    },
    })

    # Add a chart title and set the position of the legend.
    piechart.set_title({'name': 'Space analysis in Section'})
    piechart.set_legend({'position': 'right'})

    # Insert the chart into the worksheet.
    worksheet.insert_chart('G2', piechart)




    chart = workbook.add_chart({'type': 'column'})

    # Configure the series of the chart from the spreadsheet data.
    # for i in range(len(count)):
    
    chart.add_series({
        'values': f'=Sheet1!$D${2}:$D${len(counts)+1}',
        'name':   f'Count',
    })

    # Optionally, set the chart title and axis labels.
    chart.set_title({'name': 'Count of space in each sections'})
    chart.set_x_axis({'name': 'Section number'})
    chart.set_y_axis({'name': 'Count of space'})

    # Insert the chart into the worksheet.
    worksheet.insert_chart('O2', chart)








    # Close the workbook, this step is important to save your file.
    workbook.close()

def distance_contour_line(contour):
    
    longest_contour = max(contour, key=cv2.contourArea)
    shortest_contour = min(contour, key=cv2.contourArea)
    epsilon = 0.005*cv2.arcLength(shortest_contour, True)
    approx_line = cv2.approxPolyDP(shortest_contour, epsilon, True)
    
    line_length = 0
    for i in range(1, len(approx_line)):
        line_length += np.linalg.norm(approx_line[i][0] - approx_line[i-1][0])


    [vx, vy, x, y] = cv2.fitLine(longest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    

    angle_rad = np.arctan2(vy, vx)
    

    return (line_length,angle_rad)

    




def real_coordinate(point,screen,path):
    tif_path = path
    
    # Define the offset in pixels
    pixel_offset_x = point[0]
    pixel_offset_y = point[1]

    with rasterio.open(tif_path) as src:
        width = src.width
        height = src.height

        pixel_offset_x = int(width * pixel_offset_x / screen[1])
        pixel_offset_y = int(height * pixel_offset_y / screen[0])

        offset_x = src.bounds.left + pixel_offset_x * src.res[0]  # src.res[0] is the width of a pixel
        offset_y = src.bounds.top - pixel_offset_y * src.res[1]   # src.res[1] is the height of a pixel

        if src.crs.is_geographic:
            # If the image is already in geographic coordinates, just assign the values
            latitude, longitude = offset_y, offset_x
        else:
            # If not, we have to transform the coordinates from the image CRS to EPSG:4326 (lat/long)
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            longitude, latitude = transformer.transform(offset_x, offset_y)

        return (longitude,latitude)

def cv2_to_qpixmap(cv_image):
    height, width, channel = cv_image.shape
    bytes_per_line = 3 * width
    q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(q_image)

def make_kml(lines,path):
    kml = simplekml.Kml()

    # Define coordinates for the lines 
    # Each pair of tuples represents one line
    lines_coords = lines

    # Loop through the list of coordinate pairs and create a line for each
    for index, coords in enumerate(lines_coords):
        linestring = kml.newlinestring(name=f'Line {index+1}',
                                    description='A sample line',
                                    coords=coords)
        
        # Set style for the linestring
        linestring.style.linestyle.color = simplekml.Color.blue  # Blue color
        linestring.style.linestyle.width = 3  # Width of the line

    kml.save(path)

def line_intersection(line1, line2):
    
    
    

    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Line 1 represented as a1x + b1y = c1
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1
    
    # Line 2 represented as a2x + b2y = c2
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3
    a1 = np.float64(a1)
    a2 = np.float64(a2)
    b1 = np.float64(b1)
    b2 = np.float64(b2)
    c1 = np.float64(c1)
    c2 = np.float64(c2)
    determinant = a1 * b2 - a2 * b1
    
    if determinant == 0:
        # The lines are parallel; no intersection
        return None
    else:
        # Calculate the point of intersection
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return int(x), int(y)

# Function to check if a point is on a line segment
def on_segment(p, q, r):
    if max(p[0], r[0]) >= q[0] >= min(p[0], r[0]) and max(p[1], r[1]) >= q[1] >= min(p[1], r[1]):
        return True
    return False

# Check if line segments p1q1 and p2q2 intersect
def segments_intersect(p1, q1, p2, q2):
    intersection = line_intersection((p1[0], p1[1], q1[0], q1[1]), (p2[0], p2[1], q2[0], q2[1]))
    if intersection is None:
        return None
    if on_segment(p1, intersection, q1) and on_segment(p2, intersection, q2):
        return intersection
    return None
def get_crossPoint(line_segment_start,line_segment_end,contour):


    for i in range(len(contour)):
        pt1 = tuple(contour[i][0])
        pt2 = tuple(contour[(i + 1) % len(contour)][0]) 
 
        intersection = segments_intersect(line_segment_start, line_segment_end, pt1, pt2)
        if intersection:
            break  # Remove 'break' if you want to find all intersections
    return intersection


def contour_include(contour_outer, contour_inner):

    for point in contour_inner:
        # Ensure the point is converted to a tuple of integers (or floats if necessary)
        x, y = point.ravel()
        test_point = (int(x), int(y))  # or use float(x), float(y) if coordinates are floating-point numbers

        # Now use the correctly formatted `test_point` in `pointPolygonTest`.
        if cv2.pointPolygonTest(contour_outer, test_point, False) < 0:
            return False
            
    return True

def distance_Between(start,end,point):
    dx_line = end[0] - start[0]
    dy_line = end[1] - start[1]
    normal_vector = np.array([-dy_line, dx_line])
    point_vector = np.array([point[0] - start[0], point[1] - start[1]])
    distance = np.abs(np.dot(normal_vector, point_vector)) / np.linalg.norm(normal_vector)
    return int(distance)

def sort_contour(contour, start,end):
    x, y, w, h = cv2.boundingRect(contour)            
    center = (int(x + w / 2), int(y + h / 2))

    return distance_Between(start, end, center)

class ProgressDialog:
    def __init__(self, parent):
        self.top = Toplevel(parent)
        self.top.title("Progress")
        self.label = customtkinter.CTkLabel(self.top, text="Waiting...")
        self.label.pack(pady=10)

        self.progressbar = ttk.Progressbar(self.top, orient="horizontal", length=300, mode='determinate')
        self.progressbar.pack(padx=10, pady=10)
        self.progressbar["value"] = 0
        self.progressbar["maximum"] = 100

        self.cancel_button = customtkinter.CTkButton(self.top, text="Cancel", command=self.cancel)
        self.cancel_button.pack(pady=10)

    def set_value(self, value):
        self.progressbar["value"] = value
        if value >= self.progressbar["maximum"]:
            self.close()

    def cancel(self):
        # Implement what should happen when you hit cancel
        print("Cancelled by user.")
        self.close()

    def close(self):
        self.top.destroy()

class App(customtkinter.CTk):
    file_path = ""
    file_path_tif = ""
    line_list = []
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Ahau-X® RSG")
        self.geometry(f"{1400}x{800}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width= 300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.open_tif_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=10)
        self.open_tif_frame.grid(row=0, column=0,padx=10, pady=10)
        self.open_tif_button = customtkinter.CTkButton(self.open_tif_frame, text='Open TIF', command=self.open_tif)
        self.open_tif_button.grid(row=1, column=0, padx=20, pady=10)

        self.tif_path = customtkinter.CTkEntry(self.open_tif_frame,placeholder_text="Tif file path")
        self.tif_path.grid(row=2, column=0, padx=20, pady=10)


        self.convert_tif_button = customtkinter.CTkButton(self.open_tif_frame, text='Convert', command=self.convert_tif)
        self.convert_tif_button.grid(row=3, column=0, padx=20, pady=10)

        
        self.open_image_button = customtkinter.CTkButton(self.sidebar_frame, text = 'Open image',width=200, command=self.open_image)
        self.open_image_button.grid(row=2, column=0, padx=20, pady=10)


        self.control_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius = 10)
        self.control_frame.grid(row=3, column=0, padx=10, pady=10)
        self.control_label = customtkinter.CTkLabel(self.control_frame,text='Control variable', font=("Helvetica", 16, "bold"))
        self.control_label.grid(row = 0, column = 0 , columnspan =2, padx =10,pady =10)
        self.control_width_label = customtkinter.CTkLabel(self.control_frame, text='Width :')
        self.control_width_label.grid(row = 1,column = 0, padx=(10,0))
        self.control_width_text = customtkinter.CTkEntry(self.control_frame)
        self.control_width_text.grid(row = 1,column = 1, padx = 10, pady =10)
        self.control_width_text.insert(0,'1.4')

        self.control_space_label = customtkinter.CTkLabel(self.control_frame, text='Space :')
        self.control_space_label.grid(row = 2,column = 0, padx=(10,0))
        self.control_space_text = customtkinter.CTkEntry(self.control_frame)
        self.control_space_text.grid(row = 2,column = 1, padx = 10, pady =10)
        self.control_space_text.insert(0,'1.2')
        
        self.control_distance_label = customtkinter.CTkLabel(self.control_frame, text='Average width:')
        self.control_distance_label.grid(row = 3,column = 0, padx=(10,0))
        self.control_distance_slider = customtkinter.CTkSlider(self.control_frame, from_=1, to=12, number_of_steps=32)
        self.control_distance_slider.grid(row=3, column=1, padx=(20, 10), pady=(10, 10))
        self.control_distance_slider.set(12)

        self.control_distance_label = customtkinter.CTkLabel(self.control_frame, text='Deviation plant:')
        self.control_distance_label.grid(row = 4,column = 0, padx=(10,0))
        self.control_deviation_slider = customtkinter.CTkSlider(self.control_frame, from_=1, to=12, number_of_steps=32)
        self.control_deviation_slider.grid(row=4, column=1, padx=(20, 10), pady=(10, 10))
        self.control_deviation_slider.set(1)

        self.min_plant_label = customtkinter.CTkLabel(self.control_frame, text='Plant minimum size:')
        self.min_plant_label.grid(row = 5,column = 0, padx=(10,0))
        self.min_plant_slider = customtkinter.CTkSlider(self.control_frame, from_=1, to=100, number_of_steps=32)
        self.min_plant_slider.grid(row=5, column=1, padx=(20, 10), pady=(10, 10))
        self.min_plant_slider.set(10)


        self.analysis_button = customtkinter.CTkButton(self.control_frame, text = 'Analysis', width=200, command=self.draw_image)
        self.analysis_button.grid(row=6, column=0,columnspan=2, padx=20, pady=10)



        self.ui_control_frame = customtkinter.CTkFrame(self.sidebar_frame)
        self.ui_control_frame.grid(row=5, column=0, padx=20, pady=(10, 20))


        self.appearance_mode_label = customtkinter.CTkLabel(self.ui_control_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=0, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.ui_control_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.ui_control_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=2, column=0, padx=20, pady=(0, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.ui_control_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=3, column=0, padx=20, pady=(10, 20))




       
        self.image_path = customtkinter.CTkEntry(self, placeholder_text="Image path")
        self.image_path.grid(row=3, column=1, padx=(20, 0), pady=(20, 20), sticky="nsew")
        self.save_image_button = customtkinter.CTkButton(master=self,text = 'Save image',width=100, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"),command=self.save_image)
        self.save_image_button.grid(row=3, column=2, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.make_kml_button = customtkinter.CTkButton(master=self,text = 'Make KML', width=100, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"),command=self.make_kml)
        self.make_kml_button.grid(row=3, column=3, padx=(5, 5), pady=(20, 20), sticky="nsew")


     
        self.image_frame = customtkinter.CTkFrame(self)
        self.image_frame.grid(row=0, rowspan = 3, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.image_frame.grid_columnconfigure(0,weight = 1)
        self.image_frame.grid_rowconfigure(0,weight = 1)
        self.image_frame.configure(fg_color="#bfbfbf", corner_radius=10)
        self.image_canvas = customtkinter.CTkCanvas(self.image_frame)
        self.image_canvas.grid(row=0,column=0,padx =10, pady=10,sticky="nsew")

      
        v_scrollbar = customtkinter.CTkScrollbar(self.image_frame, command=self.image_canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")

      
        h_scrollbar = customtkinter.CTkScrollbar(self.image_frame, orientation="horizontal", command=self.image_canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)


     
        self.tabview = customtkinter.CTkTabview(self, width=300)
        self.tabview.grid(row=0,rowspan = 3, column=2,columnspan = 2, padx=(20, 20), pady=(10, 10), sticky="nsew")
        self.tabview.add("All data analysis")
        self.tabview.add("Sampling data analysis")
        self.tabview.tab("All data analysis").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("Sampling data analysis").grid_columnconfigure(0, weight=1)

        self.scrollable_frame = customtkinter.CTkScrollableFrame(self.tabview.tab("All data analysis"), label_text="Analysis of each section",width=400,border_width=2)
        self.scrollable_frame.grid(row=0, column=0,padx = 30, pady=(20, 0), sticky="nsew")
        self.scrollable_frame.grid_columnconfigure((0,1,2,3),weight=1)
        

 
        self.total_result_frame = customtkinter.CTkFrame(self.tabview.tab("All data analysis"),border_width=2)
        self.total_result_frame.grid(row=1,column=0,padx = 30, pady=(20, 0), sticky="nsew")
        self.total_result_frame.grid_columnconfigure((0,1,2,3,4),weight=1)

        self.section = customtkinter.CTkLabel(self.total_result_frame,width=50, text=f"Total")
        self.section.grid(row=0, column=0,pady=5)
        self.row_length = customtkinter.CTkLabel(self.total_result_frame, width=50, text='')
        self.row_length.grid(row=0, column=1,pady=5)   
        self.space_length = customtkinter.CTkLabel(self.total_result_frame, width=50, text='')
        self.space_length.grid(row=0, column=2,pady=5)
        self.count = customtkinter.CTkLabel(self.total_result_frame, width=50, text='')
        self.count.grid(row=0, column=3,pady=5) 

        self.percentage = customtkinter.CTkLabel(self.total_result_frame, width=50, text='')
        self.percentage.grid(row=0, column=4,pady=5) 













        self.scrollable_frame1 = customtkinter.CTkScrollableFrame(self.tabview.tab("Sampling data analysis"), label_text="Analysis of each section",width=400,border_width=2)
        self.scrollable_frame1.grid(row=0, column=0,padx = 30, pady=(20, 0), sticky="nsew")
        self.scrollable_frame1.grid_columnconfigure((0,1,2,3),weight=1)
    

 
        self.total_result_frame1 = customtkinter.CTkFrame(self.tabview.tab("Sampling data analysis"),border_width=2)
        self.total_result_frame1.grid(row=1,column=0,padx = 30, pady=(20, 0), sticky="nsew")
        self.total_result_frame1.grid_columnconfigure((0,1,2,3,4),weight=1)

        self.section1 = customtkinter.CTkLabel(self.total_result_frame1,width=50, text=f"Total")
        self.section1.grid(row=0, column=0,pady=5)
        self.row_length1 = customtkinter.CTkLabel(self.total_result_frame1, width=50, text='')
        self.row_length1.grid(row=0, column=1,pady=5)   
        self.space_length1 = customtkinter.CTkLabel(self.total_result_frame1, width=50, text='')
        self.space_length1.grid(row=0, column=2,pady=5)
        self.count1 = customtkinter.CTkLabel(self.total_result_frame1, width=50, text='')
        self.count1.grid(row=0, column=3,pady=5) 

        self.percentage1 = customtkinter.CTkLabel(self.total_result_frame1, width=50, text='')
        self.percentage1.grid(row=0, column=4,pady=5) 


        self.scaling_optionemenu.set("100%")
        

        self.status_bar_frame = customtkinter.CTkFrame(self,height=20)
        self.status_bar_frame.grid(row=6,column=0,columnspan = 4, sticky="nsew")
        self.status_bar_frame.grid_rowconfigure(0, weight=1)  # Assuming the widget is in row 0
        # self.status_bar_frame.grid_columnconfigure(0, weight=1)
        

        self.status = customtkinter.CTkLabel(self.status_bar_frame,text='',font=("Helvetica", 14))
        self.status.grid(row=0, column=0,padx = 50,sticky="nsew")

        # self.progress_bar = Progressbar(self.progress_bar_frame,mode='indeterminate')
        # self.progress_bar = customtkinter.CTkProgressBar(self.progress_bar_frame)
        # self.progress_bar.grid(row=0, column=0, sticky="nsew")
        # self.progress_bar.grid_forget()
      

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    # def sidebar_button_event(self):
    #     print("sidebar_button click")
    def open_tif(self):

        file_path = filedialog.askopenfilename(title="Open tif", filetypes=[("tif files", "*.tif")])
        
        if file_path:
            self.tif_path.insert(0,'')
            self.tif_path.insert(0,file_path)
            # You can also save the file path to an instance variable if needed
            self.file_path_tif = file_path
            
    def convert_tif(self):
        
        if self.file_path_tif =='':
            messagebox.showinfo('Warning','First open tif file!')
            return
        
        self.status.configure(text = 'Converting...')
        file_path = filedialog.asksaveasfilename(
                title="Save image", 
                defaultextension=".jpg", 
                filetypes=[("JPG files","*.jpg"),("PNG files", "*.png"),  ("All files", "*.*")])


        if file_path:
          

            self.create_thumbnail_geospatial(self.file_path_tif, file_path)
            self.status.configure(text = '')
      
         
       
    def open_image(self):
        self.status.configure(text = 'Opening image...')
        # file_dialog = QFileDialog()
        # file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        file_path = filedialog.askopenfilename(
                title="Open image",  
                filetypes=[("JPG files","*.jpg"),("PNG files", "*.png"),  ("All files", "*.*")])

        if file_path:

            self.file_path = file_path
            original_image = Image.open(file_path)

            # Get aspect ratio of original image
            aspect_ratio = original_image.width / original_image.height

            # Define desired size (could also be dynamically determined by some criteria)
            if aspect_ratio>=1:
                desired_width = 800
                desired_height = int(desired_width / aspect_ratio)
            else:
                desired_height = 800
                desired_width = int(desired_height * aspect_ratio)
            # Resize the image with the new size and maintain the aspect ratio
            scaled_image = original_image.resize((desired_width, desired_height), Image.Resampling.LANCZOS)

            # Convert the Pillow image to a PhotoImage for use in Tkinter
            photo = ImageTk.PhotoImage(scaled_image)
            
            self.image_canvas.create_image(desired_width/2, desired_height/2, image=photo)

            self.image_canvas.image = photo
            self.image_canvas.update_idletasks()
            self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))
            self.image_path.delete(0,'end')
            self.image_path.insert(0,file_path)
            self.status.configure(text = '')

    def save_image(self):
        if self.file_path =='':
            messagebox.showinfo('Warning','First open the image file!')
            return
        self.status.configure(text = 'Saving image...')
 
        file_path = filedialog.asksaveasfilename(
                title="Save image", 
                defaultextension=".jpg", 
                filetypes=[("JPG files","*.jpg"),("PNG files", "*.png"),  ("All files", "*.*")])
    
        if file_path:
            
            cv2.imwrite(file_path,self.image)
            messagebox.showinfo('Alert',f'File saved:  {file_path}')
            self.status.configure(text = '')

    def make_kml(self):
        

        if (self.line_list == []) :
            messagebox.showinfo('Warning','First analyze the image')
            return

        # file_dialog = QFileDialog()
        # file_path, _ = file_dialog.getSaveFileName(self, "Save File", "", "KML File (*.kml)")
        file_path = filedialog.asksaveasfilename(
        title="Save KML", 
        defaultextension=".kml", 
        filetypes=[("KML files","*.kml")])     
        

        if file_path:
            make_kml(self.line_list, file_path)


    def result_represent(self,sum_space,sum_row,count1,spaces,rows,space_count, width,standard_width,tab_name,frame,label_space,label_row,label_count,label_percentage):
        
        print('space',spaces)
        print('row',rows)
        print('count',space_count)


        label_count.configure(text=str(count1))
        label_percentage.configure(text = str(int(sum_space*100/sum_row)))
        label_space.configure(text = str(int(sum_space)))
        label_row.configure(text = str(int(sum_row)))

        clear_frame(frame)
        section = customtkinter.CTkLabel(frame,width=50, text=f"Section\nnumber")
        section.grid(row=0, column=0, padx=0)
        
        row_length = customtkinter.CTkLabel(frame, width=50, text="Row\nlength")
        row_length.grid(row=0, column=1,padx=0)    

        space_length = customtkinter.CTkLabel(frame, width=50, text="Space\nlength")
        space_length.grid(row=0, column=2, padx=0)

        countL = customtkinter.CTkLabel(frame, width=50, text="Space\ncount")
        countL.grid(row=0, column=3, padx=0)        
        
        percentage = customtkinter.CTkLabel(frame, width=50, text="Percentage")
        percentage.grid(row=0, column=4,padx=0)   
        
        for i in range(0,len(spaces)):

            section = customtkinter.CTkLabel(frame,width=50, text=f"{i+1}")
            section.grid(row=i+1, column=0, padx=0)
            
            row_length_label = customtkinter.CTkLabel(frame, width=50, text=str(rows[i]))
            row_length_label.grid(row=i+1, column=1,padx=0)    

            space_length_label = customtkinter.CTkLabel(frame, width=50, text=str(spaces[i]))
            space_length_label.grid(row=i+1, column=2, padx=0)
            
            space_count_label = customtkinter.CTkLabel(frame, width=50, text=str(space_count[i]))
            space_count_label.grid(row=i+1, column=3,padx=0)   

            percentage = customtkinter.CTkLabel(frame, width=50, text=str(int(spaces[i]*100/rows[i])))
            percentage.grid(row=i+1, column=4,padx=0)   


        plot_tab_view = customtkinter.CTkTabview(self.tabview.tab(tab_name))
        plot_tab_view.grid(row = 2, column = 0 )
        plot_tab_view.add("Pie chart")
        plot_tab_view.add("Bar chart")
        plot_tab_view.tab("Pie chart").grid_columnconfigure(0,weight=1)
        plot_tab_view.tab("Bar chart").grid_columnconfigure(0,weight=1)

        labels = [i+1 for i in range(0,len(spaces))]
        sizes = spaces
        def generate_n_colors(n):
            cmap = plt.cm.get_cmap('hsv', n)
            return [to_hex(cmap(i)) for i in range(n)]

        # Generate 'n' colors based on the number of sizes/labels
        colors = generate_n_colors(len(spaces))
        fig, ax = plt.subplots(figsize=(4,4))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Percentage of space for each section')

        canvas = FigureCanvasTkAgg(fig, master= plot_tab_view.tab("Pie chart"))
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0)

        categories = [f"{i}" for i in labels]
        values = spaces
        # Create figure and axes objects for the bar chart
        fig, ax = plt.subplots(figsize=(4, 4))
        # Creating the bar chart
        ax.bar(categories, values, color='skyblue')
        # Set labels (optional)
        # ax.set_ylabel('space length(m)')
        ax.set_title('Space length(m)')

        # Embedding the chart into a Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_tab_view.tab("Bar chart")) # Replace plot_tab_view with the actual parent widget
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0)













    def create_thumbnail_geospatial(self,tiff_path, jpeg_path):
        
    

        global stop_thread
        ds = gdal.Open(tiff_path, gdal.GA_ReadOnly)
        gt = ds.GetGeoTransform()

        # Downsample the image by a factor of 4, for example
        pixels = max(ds.RasterXSize,ds.RasterYSize)

        
        scale_factor = pixels / 60000
        if scale_factor < 1:
            scale_factor = 1

        downsampled_width = ds.RasterXSize // scale_factor
        downsampled_height = ds.RasterYSize // scale_factor

        output_ds = gdal.Translate(
            jpeg_path,
            ds,
            width=downsampled_width,
            height=downsampled_height,
            format='JPEG'
        )
        output_ds = None  # Close the file
        stop_thread = True       

    def draw_image(self):



        self.status.configure(text='Analyzing...')
        if self.file_path =='':
            messagebox.showinfo('Warning','First open the image!')
            return
        # to read the image and detect the plants

        if self.file_path_tif=='':
            messagebox.showinfo('Warning','First open the tif!')
            return

        path = str(self.file_path)
        try:
            image = cv2.imread(path)

        except:
            messagebox.showinfo('Warning','Limit pixel!')
            return
                

        spaces = []
        rows = []
        space_count = []

        sampling_spaces = []
        sampling_rows = []
        sampling_counts = []
        
        os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))  # Allow images up to ~1 TB (rough estimate)

        # Now it's safe to import and use OpenCV's imread function
        
       
        # self.progress_dialog = ProgressDialog(self)

        # self.progress_dialog.set_value(0)
        # self.progress_bar.grid(row=0, column=0, sticky="nsew")
        progress_percentage = 0
        

        # self.progress_bar.set(0)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        screen_height,screen_width, channels = image.shape # 
        screen_size = max(screen_height,screen_width)

        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        contours_line, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        if contours_line is None:
            messagebox.showinfo('Warning','There is no standard blue lines. Please add them.')
            return
        contours_line = sorted(contours_line, key=cv2.contourArea, reverse=True)
    
        longest_contour = contours_line[0]
        shortest_contour = contours_line[1]
        
        cv2.drawContours(image,longest_contour,-1, (0, 0, 0), 3)
        cv2.drawContours(image,shortest_contour,-1, (255, 255, 255), 3)

        epsilon = 0.005*cv2.arcLength(shortest_contour, True)
        approx_line = cv2.approxPolyDP(shortest_contour, epsilon, True)
        line_length = 0


        for i in range(1, len(approx_line)):
            line_length += np.linalg.norm(approx_line[i][0] - approx_line[i-1][0])


        [vx, vy, x, y] = cv2.fitLine(longest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        ellipse = cv2.fitEllipse(longest_contour)
 
        angle = ellipse[2]
        width = line_length / 5

        standard_width = int(float(self.control_width_text.get()))     

        limit_space = int(float(self.control_space_text.get())*width/standard_width)     
        angle_rad = np.deg2rad(angle + 90)

        angle_zero = np.deg2rad(angle)


        if angle > 90:
            zero=(0,0)
        else:
            zero=(0,screen_height) 

        end_zero_x = int(zero[0] + 100 * np.cos(angle_zero))
        end_zero_y = int(zero[1] + 100 * np.sin(angle_zero))

        print("angle->",angle)
        print("width->",width)
     
        # self.progress_dialog.set_value(5)
        
        lower_green = np.array([25, 50, 50])
        upper_green = np.array([85, 255, 255])
        # lower_green = np.array([30, 50, 50])
        # upper_green = np.array([80, 255, 255])
        # lower_green = np.array([35, 50, 50])
        # upper_green = np.array([77, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        sum_row = 0
        sum_space = 0
        real_lines = []

        width_slider = self.control_distance_slider.get()
        deviation_slider = self.control_deviation_slider.get()

        # Define the range for the red color
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
      
        mask = mask1 + mask2
        contours1, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        main_contour_list =  sorted(contours1, key=cv2.contourArea, reverse=True)
        progress = 0
        
        
        main_list = []

        region_min = math.inf
        min_contour = None
        for main in main_contour_list:
            if (cv2.contourArea(main) < 5000):
                continue
            region = cv2.contourArea(main)
            if (region_min > region):
                region_min = region
                min_contour = main

            main_list.append(main)


        x, y, w, h = cv2.boundingRect(min_contour)
        center_main_contour = (int(x + w / 2), int(y + h / 2))
        font_scale = int(min(w,h)/80)
       
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255) # Green color
        thickness = font_scale
        line_type = cv2.LINE_AA

        for main_contour in main_list:
            
            section_space_count = 0
            section_row = 0
            section_space = 0

            sampling_section_space_count = 0
            sampling_section_row = 0    
            sampling_section_space = 0


          

            cv2.drawContours(image, main_contour, -1, (255, 255, 255), 50)
            index = -1
            plants = []
            selected_row = []
        

            
            for contour in contours:

                x, y, w, h = cv2.boundingRect(contour)
                if (10000> w*h > self.min_plant_slider.get() and contour_include(main_contour,contour)):
    

                    cv2.drawContours(image,contour,-1, (0, 255, 0), 3)
                    plants.append(contour)
            
        
            if len(plants) < 1:
                messagebox.showinfo('Warning','There is no plant')
                return
            



            plants =  sorted(plants, key=lambda contour: sort_contour(contour,zero,(end_zero_x,end_zero_y)))

            flag=np.zeros(len(plants)) 
            

            repeat = 0

            for i in range(0,len(plants)-1):
                if flag[i] == 1:
                    continue
                

                x, y, w, h = cv2.boundingRect(plants[i])            
                centerI = (int(x + w / 2), int(y + h / 2))
                end_x = int(centerI[0] + screen_size * np.cos(angle_rad))
                end_y = int(centerI[1] + screen_size * np.sin(angle_rad))

                if i != 0:
                
                    close_count = 0
                    for index in range(0,len(selected_row)):
                        dis = distance_Between(centerI, (end_x,end_y),selected_row[index])
                        if dis < width * width_slider/16:
                            close_count += 1
                            continue
                    if close_count > 0:
                        flag[i] = 1
                        continue
                if (w*h) < 50:
                    continue   
                select = centerI
                select_end = (end_x, end_y)
                select_index = i
                selected_row.append(select)


                cv2.drawContours(image, plants[i],-1,(255,0,0),3)
                cross11 = get_crossPoint(centerI, (end_x,end_y),main_contour)
                cross22 = get_crossPoint(centerI,(end_x,end_y),plants[i])

                repeat += 1
                if repeat == 11:
                    repeat = 1

     
                if (cross11 and cross22):
                    distanceTwoPlants = cv2.norm(cross11, cross22)


                    if (distanceTwoPlants > 1000):
                        flag[i] = 1
                        continue

                    if int(distanceTwoPlants) > limit_space:
                        real_point1 = real_coordinate(cross11,(screen_height,screen_width),self.file_path_tif)
                        real_point2 = real_coordinate(cross22,(screen_height,screen_width),self.file_path_tif)

                        real_lines.append([real_point2,real_point1])

                        distanceTwoPlants = cv2.norm(cross11, cross22)

                        if repeat == 10:
                            sampling_section_space_count +=1
                            sampling_section_space += distanceTwoPlants 
                        
                        sum_space += distanceTwoPlants
                        count +=1
                        cv2.line(image,cross11,cross22,(0,0,255),3)
                        
                        section_space_count += 1
                        section_space += distanceTwoPlants

                    flag[i] = 1

                for j in range(i+1,len(plants)):

                    # progress_percentage += 1 / len(main_list)*len(plants)
                    # if progress_percentage > 0.99:
                    #     progress_percentage=0.99
                    
                    # self.progress_dialog.set_value(int(progress_percentage*100))
          
                    if flag[j] == 1:
                        continue
                    x, y, w, h = cv2.boundingRect(plants[j])   
                    centerJ = (int(x + w / 2), int(y + h / 2))
                    distance = distance_Between(select,select_end,centerJ)
                    main_distance = distance_Between(centerI,(end_x, end_y),centerJ)

                    if main_distance > width / 2:
                        continue

                    if (distance < (width*deviation_slider / 16)):
            

                        line_segment_start = select
                        line_segment_end = centerJ
                        cross_point1 = get_crossPoint(line_segment_start,line_segment_end,plants[j])
                        cross_point2 = get_crossPoint(line_segment_start,line_segment_end,plants[select_index])
            
                        

                
                        if cross_point1 is not None and cross_point2 is not None:
                            distanceTwoPlants = cv2.norm(cross_point1, cross_point2)
                            if distanceTwoPlants > 5000:
                                break
                            # limit_space = 0                 
                            if int(distanceTwoPlants) > limit_space:
                                real_point1 = real_coordinate(cross_point1,(screen_height,screen_width),self.file_path_tif)
                                real_point2 = real_coordinate(cross_point2,(screen_height,screen_width),self.file_path_tif)

                                real_lines.append([real_point2,real_point1])
                                sum_space += distanceTwoPlants
                                count +=1
                               

                                if repeat == 10:
                                    sampling_section_space_count +=1
                                    sampling_section_space += distanceTwoPlants        

                                section_space_count += 1
                                section_space += distanceTwoPlants
                                
                                cv2.line(image,cross_point1,cross_point2,(0,0,255),3)
                        flag[j]  = 1
                        flag[select_index] = 1
                        select = centerJ
                        select_index = j
                        select_end = (int(select[0] + 100 * np.cos(angle_rad)),int(select[1] + 100 * np.sin(angle_rad)))
                        
                end_x = int(select[0] - screen_size * np.cos(angle_rad))
                end_y = int(select[1] - screen_size * np.sin(angle_rad))   
                
                cv2.drawContours(image, plants[select_index],-1,(0,0,0),3)
                cross1 = get_crossPoint(select, (end_x,end_y),main_contour)
                cross2 = get_crossPoint(select,(end_x,end_y),plants[select_index])

                distance_row = cv2.norm(cross1,cross11)
                sum_row  += distance_row
                section_row += distance_row
                if repeat == 10:
                    sampling_section_row += distance_row
                if (cross1 and cross2):

                    distanceTwoPlants = cv2.norm(cross1, cross2)
                    
                    if int(distanceTwoPlants) > limit_space:
                    
                        real_point1 = real_coordinate(cross1,(screen_height,screen_width),self.file_path_tif)
                        real_point2 = real_coordinate(cross2,(screen_height,screen_width),self.file_path_tif)

                        real_lines.append([real_point2,real_point1])
                        
                        
                        sum_space += distanceTwoPlants
                        count +=1
                        

                        if repeat == 10:
                            sampling_section_space_count +=1
                            sampling_section_space += distanceTwoPlants 

                        section_space_count += 1
                        section_space += distanceTwoPlants

                    flag[select_index] = 1
                    cv2.line(image,cross1,cross2,(0,0,255),3)    
            

            spaces.append(int(section_space/(width/standard_width)))
            rows.append(int(section_row/(width/standard_width)))
            space_count.append(section_space_count)
          
            sampling_spaces.append(int(sampling_section_space/(width/standard_width)))
            sampling_rows.append(int(sampling_section_row/(width/standard_width)))
            sampling_counts.append(sampling_section_space_count)

            
            x, y, w, h = cv2.boundingRect(main_contour)
            position = (int(x + w / 2), int(y + h / 2))
   
          
            progress += 1

            text = str(progress)
            cv2.putText(image, text, position, font, font_scale, color, thickness, line_type)
   
        

        self.result_represent(sum(spaces),sum(rows),sum(space_count),spaces,rows,space_count, width,standard_width,'All data analysis',self.scrollable_frame,self.space_length,self.row_length,self.count,self.percentage)

        self.result_represent(sum(sampling_spaces),sum(sampling_rows),sum(sampling_counts),sampling_spaces,sampling_rows,sampling_counts, width,standard_width,'Sampling data analysis',self.scrollable_frame1,self.space_length1,self.row_length1,self.count1,self.percentage1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        self.image = image


        aspect_ratio = pil_image.width / pil_image.height

        # Define desired size (could also be dynamically determined by some criteria)
        if aspect_ratio>=1:
            desired_width = 800
            desired_height = int(desired_width / aspect_ratio)
        else:
            desired_height = 800
            desired_width = int(desired_height * aspect_ratio)
        # Resize the image with the new size and maintain the aspect ratio
        scaled_image = pil_image.resize((desired_width, desired_height), Image.Resampling.LANCZOS)

        # Convert the Pillow image to a PhotoImage for use in Tkinter
        photo = ImageTk.PhotoImage(scaled_image)
        
        self.image_canvas.create_image(desired_width/2, desired_height/2, image=photo)

        self.image_canvas.image = photo

        self.image_canvas.update_idletasks()
        self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))
        






        self.line_list = real_lines  




        add_excel(spaces,rows,space_count,'all')
        add_excel(sampling_spaces,sampling_rows,sampling_counts,'sampling')
               
        # self.progress_dialog.set_value(100)
        self.status.configure(text='')

        # self.progress_bar.set(1)
        # self.progress_bar.grid_forget()
     
     


if __name__ == "__main__":
    app = App()
    app.mainloop()