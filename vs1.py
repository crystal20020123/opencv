import sys
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


from osgeo import gdal

def create_thumbnail_geospatial(tiff_path, jpeg_path):
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



def convert_tiff_to_jpg(tiff_filename, jpg_filename):
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(tiff_filename) as img:
        # Convert the image to RGB mode
        width, height = img.size    
        print(width, height)        
        # img.thumbnail((img.width // 4, img.height // 4))
        img = img.resize((width // 2, height // 2), Image.Resampling.LANCZOS)
        print('ok')
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(jpg_filename, "JPEG", quality = 80)


def add_excel(spaces, rows, count):
    

    # Create a new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook('statistic.xlsx')
    worksheet = workbook.add_worksheet()

    # Some sample data for the pie chart.
    data = [
        ['Section', 'Spacelengh','Rowlength','SpaceCount','Percentage'],
    ]

    total_space=0
    total_row=0
    total_count=0
    for i in range(len(spaces)):
        list = []
        list.append('Section '+ str(i+1))
        list.append(spaces[i])
        list.append(rows[i])
        list.append(count[i])
        list.append(int(spaces[i]*100/rows[i]))
        data.append(list)
        total_row += rows[i]
        total_space += spaces[i]
        total_count += count[i]
    list = ['Total',total_space,total_row,total_count,int(total_space*100/total_row)]
    data.append(list)

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
        'values': f'=Sheet1!$D${2}:$D${len(count)+1}',
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


class HoverButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
    
    def enterEvent(self, event):
        self.setStyleSheet("background-color: #1a1a1a;font-size:20px; color: white; font-weight: bold;border: 1px solid #595959; border-radius: 5px; padding: 5px;")
    
    def leaveEvent(self, event):
        self.setStyleSheet("background-color: #bfbfbf;font-size:20px; border: 1px solid #595959; border-radius: 5px; padding: 5px;")


class ImageViewer(QWidget):
    file_path = ""
    file_path_tif = ""
    line_list = []

    image = ()
    def __init__(self):
        self.file_path = ""
        self.file_path_tif = ""
        self.line_list = []
     
        super().__init__()

        self.setWindowTitle("space analysis")
        # self.setFixedSize(1000,800)

        
        layout = QVBoxLayout()



        layoutH8 = QHBoxLayout()
        self.open_button_tif = HoverButton("Open TIF")
        self.open_button_tif.setStyleSheet("background-color: #bfbfbf;font-size:20px; border: 1px solid #595959; border-radius: 5px; padding: 5px;")
        self.open_button_tif.clicked.connect(self.open_tif)
        # self.open_button_tif.setFixedWidth(100)

        self.label10 = QLabel("FilePath : ")
        self.label10.setStyleSheet("color: #1a1a1a; font-size: 20px; padding: 10px;")
        self.label10.setFixedWidth(500)
        

        self.convert_button = HoverButton("Convert to JPG")
        self.convert_button.setStyleSheet("background-color: #bfbfbf;font-size:20px; border: 1px solid #595959; border-radius: 5px; padding: 5px;")
        self.convert_button.clicked.connect(self.convert_tif)
        # self.convert_button.setFixedWidth(200)

       
        
        layoutH8.addWidget(self.open_button_tif)
        layoutH8.addWidget(self.label10)
        layoutH8.addWidget(self.convert_button)
        layout.addLayout(layoutH8)

        self.image_label = QLabel()
        self.image_label.setStyleSheet("background-color: #bfbfbf;border-radius: 10px;")
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.image_label)
        # self.scroll.setFixedHeight(500)
        layout.addWidget(self.scroll)


        layoutH5 = QHBoxLayout()

        self.open_button = HoverButton("Open Image")
        self.open_button.setStyleSheet("background-color: #bfbfbf;font-size:20px; border: 1px solid #595959; border-radius: 5px; padding: 5px;")
        self.open_button.clicked.connect(self.open_image)
        self.open_button.setFixedHeight(50)
        layoutH5.addWidget(self.open_button)
        
        self.save_button = HoverButton("Save Image")
        self.save_button.setStyleSheet("background-color: #bfbfbf;font-size:20px; border: 1px solid #595959; border-radius: 5px; padding: 5px;")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setFixedHeight(50)
        layoutH5.addWidget(self.save_button)



        
        layout.addLayout(layoutH5)


       
        layoutH= QHBoxLayout()

        
        self.label1 = QLabel("Space :")
        self.label1.setStyleSheet("color: #1a1a1a; font-size: 30px; padding: 10px;")
        # self.label1.setFixedWidth(200)
        self.label11 = QLabel("Width :")
        self.label11.setStyleSheet("color: #1a1a1a; font-size: 30px; padding: 10px;")
        # self.label11.setFixedWidth(200)
        layoutH.addWidget(self.label1)
  


        self.text_edit = QTextEdit()
        self.text_edit.setFixedHeight(50)
        self.text_edit.setText("1.2")
        self.text_edit.setStyleSheet("background-color: white; border-radius: 5px; font-weight: bold;color: black; font-size: 20px; border: 1px solid gray; padding: 5px;")

        self.text_edit1 = QTextEdit()
        self.text_edit1.setFixedHeight(50)
        self.text_edit1.setText("1.4")
        self.text_edit1.setStyleSheet("background-color: white; border-radius: 5px; font-weight: bold;color: black; font-size: 20px; border: 1px solid gray; padding: 5px;")

        layoutH.addWidget(self.text_edit)
        layoutH.addWidget(self.label11)
        layoutH.addWidget(self.text_edit1)





        layout.addLayout(layoutH)


        layoutH2 = QHBoxLayout()
        
        self.label2 = QLabel("       The Count of spaces :")
        self.label2.setStyleSheet("color: #1a1a1a; font-size: 30px; padding: 10px;")
        layoutH2.addWidget(self.label2)

        self.label3 = QLabel("")
        self.label3.setStyleSheet("background-color: #bfbfbf; color: #1a1a1a;border-radius: 5px; font-size: 30px; padding: 10px;")
        self.label3.setFixedSize(500,50)
        layoutH2.addWidget(self.label3)

        layout.addLayout(layoutH2)
        

        layoutH3 = QHBoxLayout()
        
        self.label4 = QLabel("       The percentage of spaces :")
        self.label4.setStyleSheet("color: #1a1a1a; font-size: 30px; padding: 10px;")
        layoutH3.addWidget(self.label4)

        self.label5 = QLabel("")
        self.label5.setStyleSheet("background-color: #bfbfbf; color: #1a1a1a;border-radius: 5px; font-size: 30px; padding: 10px;")
        self.label5.setFixedSize(500,50)
        layoutH3.addWidget(self.label5)

        layout.addLayout(layoutH3)
      
        layoutH6=QHBoxLayout()

        self.label6 = QLabel("       The total length of spaces :")
        self.label6.setStyleSheet("color: #1a1a1a; font-size: 30px; padding: 10px;")
        layoutH6.addWidget(self.label6)

        self.label7 = QLabel("")
        self.label7.setStyleSheet("background-color: #bfbfbf; border-radius: 5px;color: #1a1a1a; font-size: 30px; padding: 10px;")
        self.label7.setFixedSize(500,50)
        layoutH6.addWidget(self.label7)

        layout.addLayout(layoutH6)

        layoutH7=QHBoxLayout()

        self.label8 = QLabel("       The total length of rows :")
        self.label8.setStyleSheet("color: #1a1a1a; font-size: 30px; padding: 10px;")
        layoutH7.addWidget(self.label8)

        self.label9 = QLabel("")
        self.label9.setStyleSheet("background-color: #bfbfbf; border-radius: 5px;color: #1a1a1a; font-size: 30px; padding: 10px;")
        self.label9.setFixedSize(500,50)
        layoutH7.addWidget(self.label9)

        layout.addLayout(layoutH7)



        layoutH9 = QHBoxLayout()

        self.button = HoverButton("Make the image")
        self.button.clicked.connect(self.draw_image)
        self.button.setFixedHeight(50)
        self.button.setStyleSheet("background-color: #bfbfbf;font-size:20px; border: 1px solid #595959; border-radius: 5px; padding: 5px;")
        
        self.button1 = HoverButton("Make KML")
        # self.button1.clicked.connect(self.draw_image)
        self.button1.setFixedHeight(50)
        self.button1.clicked.connect(self.make_kml)
        
        self.button1.setStyleSheet("background-color: #bfbfbf;font-size:20px; border: 1px solid #595959; border-radius: 5px; padding: 5px;")
        layoutH9.addWidget(self.button)       
        layoutH9.addWidget(self.button1)

        layout.addLayout(layoutH9)
        

        self.setLayout(layout)
  
    def save_image(self):
        if self.file_path =='':
            QMessageBox.warning(self, "Warning", "Open the image file.")
            return

        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save File", "", "Image Files (*.png *.jpg *.jpeg)")
        
        
        if file_path:
        
            cv2.imwrite(file_path,self.image)
            print("File saved:", file_path) 
    
    def make_kml(self):
        if (self.line_list == []) :
            QMessageBox.warning(self, "Warning", "First, Draw red lines.")
            return

        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save File", "", "KML File (*.kml)")
        
        

        if file_path:
            make_kml(self.line_list, file_path)
            
    def convert_tif(self):
        
        if self.file_path_tif =='':
            QMessageBox.warning(self, "Warning", "Open the tif file.")
            return


        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save File", "", "Image Files (*.png *.jpg *.jpeg)")
        
        if file_path:
            # convert_tiff_to_jpg(self.file_path_tif, file_path)
            create_thumbnail_geospatial(self.file_path_tif, file_path)
            # gdal_convert(self.file_path_tif, file_path)



    def open_tif(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open tif", "", "tif File (*.tif)")
        if file_path:
            self.label10.setText('FilePath : ')
            self.label10.setText(self.label10.text() + str(file_path))
            self.file_path_tif = file_path
    def open_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(500, 5000, aspectRatioMode=True))
            self.image_label.setAlignment(QtCore.Qt.AlignCenter)
            self.file_path = file_path
        
    def draw_image(self):




        if self.file_path =='':
            QMessageBox.warning(self, "Warning", "Open the image.")
            return
        # to read the image and detect the plants

        if self.file_path_tif=='':
            QMessageBox.warning(self, "Warning", "First open tif.")
            return

        path = str(self.file_path)
        try:
            image = cv2.imread(path)

        except:
            QMessageBox.warning(self, "Warning", "LIMIT PIXEL")
            return
                

        spaces = []
        rows = []
        space_count = []
       
        
        os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))  # Allow images up to ~1 TB (rough estimate)

        # Now it's safe to import and use OpenCV's imread function
        
       
        progressDialog = QProgressDialog("Waiting...", "Cancel", 0, 100, self)
        progressDialog.setWindowModality(Qt.WindowModal)     
        progressDialog.setValue(0)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        screen_height,screen_width, channels = image.shape # 
        screen_size = max(screen_height,screen_width)

        lower_blue = np.array([115,50,50])
        upper_blue = np.array([125,255,255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        contours_line, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        if contours_line is None:
            QMessageBox.warning(self, "Warning", "There is no standard blue lines. Please add them.")
            return

        contours_line = sorted(contours_line, key=cv2.contourArea, reverse=True)
        print(len(contours_line))

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

        standard_width = int(float(self.text_edit1.toPlainText()))     

        limit_space = int(float(self.text_edit.toPlainText())*width/standard_width)     
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
     
        # self.image = image
        # return
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

        # Define the range for the red color
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        # progressDialog.setValue(10)
      
        mask = mask1 + mask2

        
        # Find contours in the mask
        contours1, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assuming that the largest contour by area is the main region
        # main_contour = contours[]



        main_contour_list =  sorted(contours1, key=cv2.contourArea, reverse=True)

       
        # main_contour = max(contours1, key=cv2.contourArea)

        # print(len(main_contour))

        # Draw the main contour on the original image (yellow outline)
 



        
        
        progress = 0
        
        
        main_list = []
        for main in main_contour_list:
            if (cv2.contourArea(main) < 5000):
                continue
            main_list.append(main)
        
       
        for main_contour in main_list:
            
            section_space_count = 0
            section_row = 0
            section_space = 0

            progress += 1

            progressDialog.setValue(int(progress * 100 / len(main_list)))
            cv2.drawContours(image, main_contour, -1, (0, 255, 255), 3)
            index = -1
            plants = []
            selected_row = []
        

            
            for contour in contours:

                x, y, w, h = cv2.boundingRect(contour)
                if (w*h > 1 and contour_include(main_contour,contour)):
    

                    cv2.drawContours(image,contour,-1, (0, 255, 0), 3)
                    plants.append(contour)
            
        
            if len(plants) < 1:
                QMessageBox.warning(self, "Warning", "There is no plant")
                return

            
            



            plants =  sorted(plants, key=lambda contour: sort_contour(contour,zero,(end_zero_x,end_zero_y)))

            flag=np.zeros(len(plants)) 
            
            for i in range(0,len(plants)-1):
                if flag[i] == 1:
                    continue
             
                x, y, w, h = cv2.boundingRect(plants[i])            
                centerI = (int(x + w / 2), int(y + h / 2))
                end_x = int(centerI[0] + screen_size * np.cos(angle_rad))
                end_y = int(centerI[1] + screen_size * np.sin(angle_rad))

                if i != 0:
                
                    row_count = 0
                    for index in range(0,len(selected_row)):
                        dis = distance_Between(centerI, (end_x,end_y),selected_row[index])
                        if dis < width  / 2:
                            row_count += 1
                            continue
                    if row_count > 0:
                        continue

                
                select = centerI
                select_end = (end_x, end_y)
                select_index = i
                selected_row.append(select)
                cv2.drawContours(image, plants[i],-1,(255,255,255),3)
                cross11 = get_crossPoint(centerI, (end_x,end_y),main_contour)
                cross22 = get_crossPoint(centerI,(end_x,end_y),plants[i])



        
                # progressDialog.setValue(int(i*100/len(plants)))
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


                        sum_space += distanceTwoPlants
                        count +=1
                        cv2.line(image,cross11,cross22,(0,0,255),3)
                        
                        section_space_count += 1
                        section_space += distanceTwoPlants

                    flag[i] = 1

                for j in range(0,len(plants)):
                    if flag[j] == 1:
                        continue
                    x, y, w, h = cv2.boundingRect(plants[j])   
                    centerJ = (int(x + w / 2), int(y + h / 2))
                    distance = distance_Between(select,select_end,centerJ)

                    if (distance < (width / 16)):
            

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
                if (cross1 and cross2):

                    distanceTwoPlants = cv2.norm(cross1, cross2)
                    
                    if int(distanceTwoPlants) > limit_space:
                    
                        real_point1 = real_coordinate(cross1,(screen_height,screen_width),self.file_path_tif)
                        real_point2 = real_coordinate(cross2,(screen_height,screen_width),self.file_path_tif)

                        real_lines.append([real_point2,real_point1])
                        
                        
                        sum_space += distanceTwoPlants
                        count +=1
                        
                        section_space_count += 1
                        section_space += distanceTwoPlants

                    flag[select_index] = 1
                    cv2.line(image,cross1,cross2,(0,0,255),3)    
            


            spaces.append(int(section_space/(width/standard_width)))
            rows.append(int(section_row/(width/standard_width)))
            space_count.append(section_space_count)
            section_space = 0
            section_row = 0 
            section_space_count = 0
            
        print(spaces)
        print(rows)
        progressDialog.setValue(100)


        self.label3.setText(str(count))
        try:
            self.label5.setText(str(int(sum_space*100/sum_row))+"%")
        except:
            pass
        self.label7.setText(str(int(sum_space/(width/standard_width)))+"m")
        self.label9.setText(str(int(sum_row/(width/standard_width)))+"m")

        pixmap = cv2_to_qpixmap(image)

        self.image_label.setPixmap(pixmap.scaled(500, 5000, aspectRatioMode=True))
        self.image = image
        self.line_list = real_lines  

        add_excel(spaces,rows,space_count)

        # progressDialog.setValue(100)
     

def main():
    app = QApplication([])
    image_viewer = ImageViewer()
    image_viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()