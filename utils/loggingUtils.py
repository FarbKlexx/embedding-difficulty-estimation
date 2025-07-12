'''
This file contains some helper functions to print values
'''

def print_X_y(X,y):
    c = 0
    for i in X:
        print(i)
        print("Dimension: " + str(len(i)))
        print("Difficulty: " + str(y[c]))
        print("-----------------------------------------------")
        c += 1
    print("\n")
    return
    