#!/bin/python
blocks = 8
dim_y = 51
dim_x = 68
for col in range(blocks): 
    for row in range(blocks): 
        y1 = row * dim_y 
        y2 = (row + 1) * dim_y
        x1 = col * dim_x 
        x2 = (col + 1) * dim_x

        print "diff crop " + str(y1) + ":" + str(y2) + "," + str(x1) + ":" + str(y2)