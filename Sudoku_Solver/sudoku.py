import numpy as np 

# This is a python code for solving sudoku. 

row = 9 
cols = 9 
def check(num  , r , c) : 
    #Check whether at a given cell you can place  num or not 
    for i in range(0, 9):
        if(i!= c and board[r][i] == num) : 
            return False 
    
    for i in range(0 , 9):
        if(i!=r and board[i][c] == num) : 
            return False
    
    # map r , c to its appropriate square 
    x = (r//3)*3 
    y = (c//3)*3 
    
    for i in range(0 , 3):
        for j in range(0 , 3):
            if(i+x == r and j+y == c) :
                continue 
            if board[i+x][j+y] == num :
                return False
            
    return True 
    pass 

#L-C-C-M Framework
def solver( r , c) :
    #Base case 
    if c == 9 :
        solver(r + 1 , 0)
        return
         
    if r == 9 :
        print(board)
        return 
       
    #Choices 
    if(board[r][c] == 0) : 
        for i in range(1 , 10):
            if check(i , r , c) == True : 
                board[r][c] = i 
                solver(r , c+1)
                board[r][c] = 0 
    else:
        if check(board[r][c] , r , c) :
            solver(r , c + 1)
    pass
    
board = np.zeros((row ,cols))
board = np.array([
  [4, 3, 0, 5, 0, 0, 0, 0, 6],
  [0, 0, 8, 0, 2, 0, 0, 1, 0],
  [0, 1, 7, 0, 0, 0, 4, 5, 0],
  [8, 6, 0, 2, 0, 0, 0, 7, 0],
  [0, 7, 0, 8, 0, 0, 2, 6, 0],
  [1, 0, 0, 0, 0, 7, 8, 3, 0],
  [2, 0, 0, 1, 7, 8, 0, 4, 3],
  [0, 0, 1, 0, 0, 0, 0, 0, 7],
  [0, 0, 0, 6, 5, 0, 0, 0, 0]
])

solver(0 , 0)

