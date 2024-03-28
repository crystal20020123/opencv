import tkinter as tk
import customtkinter as ctk

class CTkRangeSlider(ctk.CTkFrame):
    def __init__(self, parent, from_=0, to_=100, **kwargs):
        super().__init__(parent, **kwargs)
        self.from_ = from_
        self.to_ = to_
        self.width = kwargs.get('width', 200)  # Set a default width if not supplied

        self.slider1_pos = from_
        self.slider2_pos = to_

        # Create the sliders as CTkSlider objects
        self.slider1 = ctk.CTkSlider(self, from_=from_, to=to_)
        self.slider1.pack(fill='x', expand=True)

        self.slider2 = ctk.CTkSlider(self, from_=from_, to=to_)
        self.slider2.pack(fill='x', expand=True)

        # Bind commands to update positions
        self.slider1.configure(command=self.update_slider1_pos)
        self.slider2.configure(command=self.update_slider2_pos)

    def update_slider1_pos(self, value):
        self.slider1_pos = value
        if self.slider1_pos > self.slider2_pos:
            self.slider2.set(self.slider1_pos)

    def update_slider2_pos(self, value):
        self.slider2_pos = value
        if self.slider2_pos < self.slider1_pos:
            self.slider1.set(self.slider2_pos)

# Usage:
root = ctk.CTk()
root.geometry("300x200")

range_slider = CTkRangeSlider(root, from_=1, to_=12)
range_slider.pack(pady=20, padx=10)

def print_values():
    print(f"Value 1: {range_slider.slider1_pos}, Value 2: {range_slider.slider2_pos}")

button = ctk.CTkButton(root, text="Print Values", command=print_values)
button.pack()

root.mainloop()