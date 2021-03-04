from collections import namedtuple
from CoraData import CoraData
Data=namedtuple("Data",['x','y','adjacency','train_mask','val_mask','test_mask'])

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    cora=CoraData()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
