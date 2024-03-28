import tkinter as tk
from PIL import Image, ImageTk, ImageSequence
import time
import vs2

class CustomModal(tk.Toplevel):
    flag = vs2.stop_thread   
    def __init__(self, parent, title, message):
        super().__init__(parent)
    
        flag = vs2.stop_thread    
        # Remove window decorations
        self.overrideredirect(True)
        
        # Set the modal's position on the screen (centered over the parent)
        x = parent.winfo_rootx() + parent.winfo_width() // 2
        y = parent.winfo_rooty() + parent.winfo_height() // 2
        self.geometry(f"+{x - 200}+{y - 100}")

        # content_frame = tk.Frame(self)
        
        # content_frame.pack(padx=10, pady=10)
        # self.wm_attributes('-alpha', 0.5) 
        video_path = "loading.gif"  # Replace with the path to your GIF
        parent_bg_color = self.cget('bg') 
        # Load the GIF using PIL
        self.sequence = [ImageTk.PhotoImage(img) for img in ImageSequence.Iterator(Image.open(video_path))]
        self.canvas = tk.Canvas(self, width=200, height=200)
        self.canvas.pack()
        self.image = self.canvas.create_image(100, 100, image=self.sequence[0])

        # Close event
        self.protocol("WM_DELETE_WINDOW", self.destroy)
      
        # Animate GIF
        self.animate(1)
        
        # Take focus and make modal to the parent window
        self.grab_set()
        self.focus_set()
        self.wait_window()
    
    def animate(self, counter):
        if self.flag == True:
            self.destroy()
            
        self.canvas.itemconfig(self.image, image=self.sequence[counter])
        # Loop the animation
        counter += 1
        if counter == len(self.sequence):
            counter = 0
        self.after(50, lambda: self.animate(counter))