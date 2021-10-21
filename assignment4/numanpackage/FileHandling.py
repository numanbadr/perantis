count = 1

def create(file_name):
    f = open(f'{file_name}.txt','w')
    f.close()

def write(file_name):
    lines = int(input('How many lines? >>'))
    f = open(f'{file_name}.txt', 'w')
    
    for i in range(lines):
        f.writelines(input(f'Input message for line {i+1}: '))
        f.writelines('\n')
    f.close()
    
def read(file_name):
    f = open(f'{file_name}.txt', 'r')
    for l in f:
        print(l)
    f.close()
                     
