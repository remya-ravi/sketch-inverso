from tkinter import Tk, Button, Label
import tkinter.font as font
from tkinter.filedialog import askopenfilename
from sketchback import sketchback
path = ""
def set_path():
    global path
    path = askopenfilename()

def make_img():
    sketchback(path,"weights_faces")

    

root = Tk()
root.title("Sketch Inverso")
root.geometry("450x400")
root.configure(bg="white")
myFont = font.Font(family='Helvetica', size=30, weight='bold')
lbl = Label(root, text="Sketch Inverso",bg="white")
lbl['font'] = myFont
lbl.place(x=70,y=80)
browse = Button(root, text="Browse", bg="darkgrey",fg="white",border=0,width=30,height=2,command=lambda:set_path())
browse.place(x=110,y=160)
make = Button(root, text="Make", bg="green",fg="white",border=0,width=30,height=2,command=make_img)
make.place(x=110,y=200)
root.mainloop()
