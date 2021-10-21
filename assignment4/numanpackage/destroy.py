import os

def destroy(file_name):
    if os.path.exists(f'{file_name}.txt'):
        os.remove(f'{file_name}.txt')
    else:
        print('File does not exist.')