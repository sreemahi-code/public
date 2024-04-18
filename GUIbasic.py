#PYTHON  GUI for chatbot 
from tkinter import * 
#library we are using
#parent window
root = Tk()
root.title('First Chatty')
#Dimensions for the window
root.geometry('400x500')
main_menu = Menu(root) #menu bar creation

file_menu = Menu(root)  #sub menu 1
file_menu.add_command(label='New')
file_menu.add_command(label='Save As')
file_menu.add_command(label='No change')

weover = Menu(main_menu) #sub menu 2
weover.add_command(label='Are you sure')

main_menu.add_cascade(label='Medi Purpose', menu= file_menu)
main_menu.add_command(label='Edit')
main_menu.add_cascade(label='we over', menu = weover) #know that we have to cascade to add submenu
root.config(menu=main_menu)

#Creating the chat Window
chatWindow= Text(root, bd=1, bg='purple', width = 50, height= 8)
chatWindow.place(x=6, y=6, height = 365, width = 370)

#Creating a button 
Button = Button(root, text = 'Send', bg='red', activebackground= 'green', width = 12, height = 5, font =('Arial', 20))
Button.place(x=6, y=400, height = 88, width = 120)

#Scroll Bar
scrollbar= Scrollbar(root, command= chatWindow.yview())
scrollbar.place(x=375, y= 5, height= 385)
#Create message window
messageWindow = Text(root, bg='black', width = 30, height = 4) 
messageWindow.place(x= 125, y = 400, height = 88, width = 260)
root.mainloop()
#Create message window
messageWindow = Text(root, bg='black', width = 30, height = 4) 
messageWindow.place(x= 125, y = 400, height = 88, width = 260)
root.mainloop()
