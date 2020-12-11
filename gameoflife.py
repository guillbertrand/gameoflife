import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# create board
BOARD_SIZE = (100,150)
board = np.zeros(BOARD_SIZE)

# define start pattern. ex : R-pentomino
pattern = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]])

posX = int(BOARD_SIZE[0]/2)-int(pattern.shape[0]/2)
posY = int(BOARD_SIZE[1]/2)-int(pattern.shape[1]/2)
# write pattern on board 
board[posX:(posX+pattern.shape[0]),posY:(posY+pattern.shape[1])] = np.copy(pattern)
board = board.astype(int)

# init graph
fig = plt.figure(dpi=120)    
cmap = plt.get_cmap('gray_r')
pc = plt.pcolor(board, cmap="gray_r", edgecolors='cadetblue', linewidths=0.2)
ax = plt.gca()
time_text = ax.text(1, 1,'')

kernel = np.ones((3,3), dtype=np.int8)
kernel[1,1] = 0

def animate(i):
    global board, kernel
    # count neighbours with 2D convolution
    #neigh = np.zeros(BOARD_SIZE)
    #neigh[1:-1,1:-1] = (board[:-2,:-2]  + board[:-2,1:-1] + board[:-2,2:] + 
    #                    board[1:-1,:-2] +                   board[1:-1,2:]+ 
    #                    board[2:,:-2]   + board[2:,1:-1]  + board[2:,2:]) 
    neigh = convolve2d(board, kernel, mode='same')
    print(board)
    print('')
    print(neigh)
    # alive if 3 neighbours or 2 neighbours and already alive                    
    board = np.logical_or(neigh==3,np.logical_and(board==1,neigh==2))
    board = board.astype(int)
    # refresh graph
    time_text.set_text( str(i) )
    nc = cmap(255*board.ravel()) 
    pc.update({'facecolors':nc})


# run evolution
nb_of_generation = 1200
ani = animation.FuncAnimation(fig, animate, frames=nb_of_generation, interval=20, blit=False, repeat=False)
plt.show()
