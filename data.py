import numpy as np
import cv2
import os
import random

def unsolve_add_wall(path, out_path):
    """
    Take a path pointing to a folder of solvable mazes. Makes a copy of each maze and adds a wall to render the maze unsolvable without violating the maze axioms, if possible. 
    If successful, saves the copy in the folder pointed to by out_path.

    Args:
        path (str): The path pointing to the folder to read from. 
        out_path (str): The path pointing to the folder to output to. 
    """
    #count = 0
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
            options = tuple(dfs_path_extract(img)[0])
            if options:
                coord = random.choice(options)
                img[coord[0], coord[1]] = 0
                cv2.imwrite(os.path.join(out_path, filename), img)
                print(f"Image written to {filename}.")
            else: 
                print(f"No viable tiles to place a wall on: {filename}.")
                continue
        #count += 1
        #if count > 100:
            #return

def unsolve_maintain_wall_number(path, out_path):
    """
    Take a path pointing to a folder of solvable mazes. Makes a copy of each maze and adds a wall to render the maze unsolvable without violating the maze axioms, if possible. 
    Then, removes a wall without rendering the maze solvable again and without violating the maze axioms, if possible.
    If successful, saves the copy in the folder pointed to by out_path.

    Args:
        path (str): The path pointing to the folder to read from. 
        out_path (str): The path pointing to the folder to output to. 
    """
    #count = 0
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
            new_wall, new_path = tuple(extract_path_move(img))
            if new_wall and new_path:
                img[new_wall[0], new_wall[1]] = 0
                img[new_path[0], new_path[1]] = 255
                cv2.imwrite(os.path.join(out_path, filename), img)
                print(f"Image written to {filename}.")
            elif not new_wall: 
                print(f"No viable tiles to place a wall on: {filename}.")
                continue
            elif not new_path:
                print(f"No viable tiles to convert to a path: {filename}")
        
        #count += 1
        #if count > 100:
            #return

def l1_dist(a,b):
    """
    Compute Manhattan distance between two coordinates.

    Args:
        a (int, int): The first point. 
        b (int, int): The second point.
    
    Returns:
        int: The manhattan distance between points a and b.
    """
    return np.sum(np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]))

def solvable(img):
    """
    Take a greyscale image of a maze. Perform depth first search. Return if a solution is found or not. 

    Args:
        img (numpy.ndarray): Grayscale maze image (values range from 0 to 255)
    
    Returns:
        set: A set of all the paths that can be turned into walls to render the maze unsolvable without violating the maze axioms. 
    """
    n = len(img)
    curr = (1, 0)
    curr_options = 0
    branches = []
    curr_branches = [[]]
    while True:
        curr_options = 0
        if curr == (n - 2, n - 1):
            img[img > 0] = 255
            return True
        img[curr[0], curr[1]] = 128 # mark as visited
        neighbors = [(max(curr[0] - 1, 0), curr[1]), (min(curr[0] + 1, n - 1), curr[1]), (curr[0], max(curr[1] - 1, 0)), (curr[0], min(curr[1] + 1, n - 1))]
        for neighbor in neighbors:
            if img[neighbor[0], neighbor[1]] == 255:
                branches.append(neighbor)
                curr_options += 1
        if len(branches) == 0:
            break # we've covered the whole maze
        if curr_options == 0:
            # We're at a dead end and must go back
            if len(branches) == 1: 
                curr_branches.pop() # This was our last option, so cut this branch out of the path
            curr = branches.pop() # Go back to the last branch
        elif curr_options == 1:
            curr = branches.pop() # This is a straight path, keep moving forward
        else:
            # We're at a true branch with more than one option
            curr_branches[-1].append(curr)
            curr = branches.pop() # Pick a branch
            curr_branches.append([curr]) # Start tracking the current branch
        if len(curr_branches) == 0: # We've traversed the maze
            break
    img[img > 0] = 255 # Reset the path we traced
    return False

def dfs_path_extract(img):
    """
    Take a greyscale image of a maze. Perform depth first search. Return viable wallable pixels on the true path that are not near a fork. 

    Args:
        img (numpy.ndarray): Grayscale maze image (values range from 0 to 255)
    
    Returns:
        (set, list): A tuple consisting of the set of all the paths that can be turned into walls to render the maze unsolvable without violating the maze axioms, 
            and a list of the true solution to the maze prior to the placement of any walls. (not necessarily ordered)

    """
    n = len(img)
    curr = (1, 0)
    curr_options = 0
    branches = []
    true_path = []
    curr_branches = [[]]
    last_branch = None
    branched = False # Used to prevent from marking pixels on the initial path before a branch - potentially too easy for CNN to spot
    while True:
        curr_options = 0
        if curr == (n - 2, n - 1):
            img[curr[0], curr[1]] = 128 # mark as visited
            last_branch = curr_branches[-1]
            break
        img[curr[0], curr[1]] = 128 # mark as visited
        neighbors = [(max(curr[0] - 1, 0), curr[1]), (min(curr[0] + 1, n - 1), curr[1]), (curr[0], max(curr[1] - 1, 0)), (curr[0], min(curr[1] + 1, n - 1))]
        for neighbor in neighbors:
            if img[neighbor[0], neighbor[1]] == 255:
                branches.append(neighbor)
                curr_options += 1
        if len(branches) == 0:
            break # we've covered the whole maze
        if curr_options == 0:
            # We're at a dead end and must go back
            if len(branches) == 1: 
                curr_branches.pop() # This was our last option, so cut this branch out of the path
            curr = branches.pop() # Go back to the last branch
        elif curr_options == 1:
            curr = branches.pop() # This is a straight path, keep moving forward
            if branched: # Only want to block off paths after the first branch
                curr_branches[-1].append(curr)
        else:
            # We're at a true branch with more than one option
            branched = True
            curr = branches.pop() # Pick a branch
            curr_branches.append([curr]) # Start tracking the current branch

        if len(curr_branches) == 0: # We've traversed the maze
            break
    for l in curr_branches:
        true_path.extend(l)
    result = []
    for c in true_path:
        neighbors = [(max(c[0] - 1, 0), c[1]), (min(c[0] + 1, n - 1), c[1]), (c[0], max(c[1] - 1, 0)), (c[0], min(c[1] + 1, n - 1))]
        wall_count = 0
        for neighbor in neighbors:
            if img[neighbor[0], neighbor[1]] == 0:
                wall_count += 1
        if wall_count == 2 and img[neighbors[0]] == img[neighbors[1]] and img[neighbors[2]] == img[neighbors[3]]:
            result.append(c)
    result = set(result)
    if (n - 2, n - 1) in result:
        result.remove((n - 2, n - 1)) # Don't want to block off the exit tile
    if (1, 0) in result:
        result.remove((1, 0)) # Don't want to block off the entrance either

    img[img > 0] = 255
    # Remove the path marking coloring
    notresult = set()
    if last_branch: # Only want to block off paths before the last branch right next to the exit
        notresult = notresult.union(set(last_branch))
    for tile in result: # Check that changing each viable path doesn't violate the maze axioms
        img[tile[0], tile[1]] = 0
        if not validate_maze_structure(img): 
            notresult.add(tile)
        img[tile[0], tile[1]] = 255
    result = result.difference(notresult)
    # Testing code
    #for tile in result:
        #img[tile[0], tile[1]] = 128
    #return img, result 
    return result, true_path # Return the set of possible values to block off. true_path is used in another function.

def extract_path_move(img):
    """
    Take a greyscale image of a maze. Find the solution with DFS. First, choose an arbitrary path tile that can be turned into a wall to render the maze unsolvable.
    Then, choose a random wall tile that can be turned into a path without rendering the maze solvable again. Both of these operations should not violate the maze axioms.  

    Args:
        img (numpy.ndarray): Grayscale maze image (values range from 0 to 255)
    
    Returns:
        ((int, int), (int, int)): A pair of coordinates, the first of which represents the path that will be turned into a wall, and the second the wall that will be turned into a path. 
    """
    candidates, true_path = dfs_path_extract(img)
    if candidates:
        path_to_wall = random.choice(tuple(candidates))
    else:
        path_to_wall = None
    if not path_to_wall:
        return None, None
    img[path_to_wall[0], path_to_wall[1]] = 0
    wall_to_path = extract_safe_wall(img, true_path)
    return path_to_wall, wall_to_path



def extract_safe_wall(img, true_path):
    """
    Take a greyscale image of an unsolvable maze. Return a wall that can be turned into a path without violating the maze axioms or rendering the maze solvable. 

    Args:
        img (numpy.ndarray): Grayscale maze image (values range from 0 to 255)
        true_path (list): a list of tiles that is used in the true path. 
    
    Returns:
        (int, int): A coordinate representing a wall that can be turned into a path without violating the maze axioms or rendering the maze solvable.
    """

    coords = list(np.ndindex(img[1:len(img)-1, 1:len(img)-1].shape)) # Only consider interior coordinates
    random.shuffle(coords) # Consider coordinates in random order so our choice of wall to break is random
    for coord in coords:
        if img[coord[0] + 1, coord[1] + 1] == 0:
            for t in true_path:
                if l1_dist(coord, t) < 2: # Adding paths close to the former true path is too likely to make it solvable
                    break
            else: # Not an indentation mistake - for-else statement. We're not too close to the former true path
                img[coord[0] + 1, coord[1] + 1] = 255
                if (not solvable(img)) and validate_maze_structure(img):
                    return coord[0] + 1, coord[1] + 1
                img[coord[0] + 1, coord[1] + 1] = 0 # Reset, move to next
    img[1, 0] = 255
    return None

def validate_maze_structure(img):
    """
    Given a maze, check that it "looks like" a "genuine" maze (i.e. no corner walls, 1-tile straight walls, thick walls or paths, etc.)

    Args:
        img (numpy.ndarray): Grayscale maze image (values range from 0 to 255).
    
    Returns:
        bool: Whether the maze is solvable (True) or not (False). 
    """
    n = len(img)
    img = img/255
    img = np.pad(img, 1, mode='constant', constant_values=0)
    img[2, 0] = 1
    img[n - 1, n + 1] = 1 # Extend the path into the pad
    for y in range(2, n):
        for x in range(2, n): # Only need to check the internal tiles
            sub_5x5 = img[y-2:y+3, x-2:x+3]
            cent = sub_5x5[2, 2]
            left = sub_5x5[2, 1]
            left2 = sub_5x5[2, 0]
            right = sub_5x5[2, 3]
            right2 = sub_5x5[2, 4]
            up = sub_5x5[1, 2]
            up2 = sub_5x5[0, 2]
            down = sub_5x5[3, 2]
            down2 = sub_5x5[4, 2]
            upleft = sub_5x5[1, 1]
            upright = sub_5x5[1, 3]
            downleft = sub_5x5[3, 1]
            downright = sub_5x5[3, 3]



            if (left2 == upleft == up == upright == right2 == downright == down == downleft) and (left == right) and (left != left2): # Horizontal floating 3-log
                return False
            
            if (up2 == upright == right == downright == down2 == downleft == left == upleft) and (up == down) and (up != up2): # Vertical floating 3-log
                return False

            if left == right == up == down and left != cent: # 1-tile floating wall
                return False

            # Corner or fat wall/path (neither is allowed)
            if (left == up and upleft == cent) or (up == right and upright == cent) or (right == down and downright == cent) or (down == left and downleft == cent):
                return False

            # Wall stubs
            if left == cent and left2 != left and right != cent: # 1-tile left wall/path
                return False
            if right == cent and right2 != right and left != cent: # 1-tile right wall/path
                return False
            if up == cent and up2 != up and down != cent: # 1-tile up wall/path
                return False
            if down == cent and down2 != down and up != cent: #1-tile down wall/path
                return False
            if up == cent == down and (up2 != up or down2 != down) and (left == cent or right == cent): # T-intersection stub (vertical bar)
                return False    
            if left == cent == right and (left2 != left or right2 != right) and (up == cent or down == cent): # T-intersection stub (horizontal bar)
                return False

    return True
    
def validate_mazes(path, count):
    """
    Takes in a path to a folder containing pngs of solvable mazes. Counts the number of mazes that is invalid (under the heuristic conditions defined above).
    Args:
        img (numpy.ndarray): Grayscale maze image (values range from 0 to 255).
    Returns:
        int: The number of invalid mazes considered.
    """
    false_count = 0
    c = 0
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
            x = validate_maze_structure(img)
            if not x:
                false_count += 1
        c += 1
        if c > count:
            return false_count
    return false_count

img = cv2.imread('./17/00000.png', cv2.IMREAD_GRAYSCALE)
print(solvable(img))
print(np.sum(img/255))
#a, b = extract_path_move(img)

#print(extract_path_move(img))
#print(validate_maze_structure(img))
#print(np.sum(img/255))
#print(extract_path_move(img))
#print(validate_maze_structure(img))

#cv2.imshow('test', cv2.resize(dfs_path_extract(img)[0].astype(np.uint8), (0, 0), fx=10, fy=10))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#print(validate_mazes('./17_unsolvable/', 10000))

#unsolve_add_wall('./17/', './17_unsolvable_extra_wall/')
#unsolve_maintain_wall_number('./49/', './49_unsolvable_same_wall_count/')
