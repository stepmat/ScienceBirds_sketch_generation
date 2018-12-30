
import sys
import cv2
import numpy
import copy
import scipy.misc
import itertools
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import morphology, label
from copy import deepcopy
from operator import itemgetter
from statistics import median, mean
from math import sqrt
from random import randint
from scipy.misc import toimage


#sketch analysis
computer_vision_size_threshold = 50
threshold_block_width = 5                   # minimum block width in pixels
threshold_block_height = 5                  # minimum block height in pixels

#Scaling method
scaling_method = int(sys.argv[2])           # 0=Max  1=Min  2=MidRange  3=Mean  4=Median            

#rules when picking a block type
check_overlap = 1                           # Check that no blocks overlap
height_error_allowed_overlap = 0.03         # prevents rounding errors and gives bit of flexability

check_local_stability = 0                   # Check that the structure is locally stable after each added block

check_global_stability = 1                  # Check that the structure is globally stable after each added block
check_global_stability_method = 2           # 1 is enforce global stability only for blocks currently added (doesn't take into account future blocks)
                                            # 2 is use both blocks added so far and sketch blocks for those not yet added

check_all_supporters = 1                    # Check that all supporters for a block are present (could possibly not be required if global stability checking is enforced)
required_support_amount = 0.01

check_groups = 1                            # Check that group height requirements are enforced
average_single_block_groups_heights = 1     # average the height of all single blocks in groups with other single blocks (helps with very poor drawings...)
height_error_allowed_groups = 0.05

use_similarity_grouping = 1
average_same_block_groups_heights = 1
error_percentage_shape = 0.05

check_era_relations = 0                     # Check that ERA relations hold

check_composite_block_stability = 1         # check that composite blocks are stable (local)

shift_blocks_sideways = 1                   # Makes structures more likely to pass but takes longer, Helps with making structures stable/no overlap
moves_to_try = [-0.8,0.7,-0.6,0.5,-0.4,0.3,-0.2,0.1]
# Alternative horizontal distance options:
#-0.4,0.35,-0.3,0.25,-0.2,0.15,-0.1,0.05
#-2.8,2.6,-2.4,2.2,-2.0,1.8,-1.6,1.4,-1.2,1.0,-0.8,0.6,-0.4,0.2
#-1.4,1.3,-1.2,1.1,-1.0,0.9,-0.8,0.7,-0.6,0.5,-0.4,0.3,-0.2,0.1

order_blocks_smart = 1                      # re-order blocks into a more logical order based on support graph (rather than simply bottom to top)

#add extra blocks into sketch
add_extra_blocks_to_make_stable = 1         # add extra blocks to sketch to make structure globally stable
push_back_distance = 5                      # distance to push extra blocks inwards (in pixels), helps deal with minor imperfections in the sketches / vision

#generate composite blocks
composite_blocks_allowed = 1                # composite blocks are allowed within the structure
rearrange_special_block_order = 1           # also include rearrangements of composite blocks as alternative options
max_composite_block_width = 3               # number of blocks wide that a composite block can be
composite_block_interweaving = 1
composite_block_penalty_picking = 1.5       # error difference when picking block type multiplied by this value if it is composite
composite_block_penalty_end = 0.0           # final error value score multiplied by this times ratio of composite blocks to non-composite blocks (NOT CURRENTLY INCLUDED)

limit_number_block_type_changes = 1         # limit the number of times a block type can change before rolling back a block
max_number_block_type_changes = 20          # increasing will give better final results but dramatically increases generation time, when using composite blocks


# SHOULD ONLY BE USED ON ACCURATE STRUCTURES WITH ORTHOGONAL/RECTILINEAR POLYGONS
corner_splitting_allowed = int(sys.argv[3]) # split polygons into rectangles based on their corners

seperate_vision_corners = 1                 # 0 = associate each corner with the MBR that it is within (problem if within two or more MBRs)
                                            # 1 = associte each corner with the MBR whose original shape it is closest too
max_distance_allowed = 3000                 # maximum distance a corner can be from an MBR (removes dots) ERROR WHEN STRUCTURE HAS HOLE IN IT!!!

corner_detection_quality_threshold = 0.2    # quality of corner required for detection
corner_detection_min_distance = 20          # minimum ditance between corners (euclidean)

threshold_corner_amount_x = 10              # make x values for corners same if wihtin this pixel distance
threshold_corner_amount_y = 10              # make y values for corners same if wihtin this pixel distance
add_togethor_similar_x = 1                  # combines groups if they share a block
add_togethor_similar_y = 1 

GARY_INITIAL = 1
MATTHEW_INITIAL = 1
OTHER_INITIAL = 1

ground = -3.5                               # position of the level ground

# blocks number and size
blocks = {'1':[0.84,0.84], '2':[0.85,0.43], '3':[0.43,0.85], '4':[0.43,0.43],
          '5':[0.22,0.22], '6':[0.43,0.22], '7':[0.22,0.43], '8':[0.85,0.22],
          '9':[0.22,0.85], '10':[1.68,0.22], '11':[0.22,1.68],
          '12':[2.06,0.22], '13':[0.22,2.06]}

original_number_blocks = len(blocks)

# blocks number and name
# (blocks 3, 7, 9, 11 and 13) are their respective block names rotated 90 derees clockwise
block_names = {'1':"SquareHole", '2':"RectFat", '3':"RectFat", '4':"SquareSmall",
               '5':"SquareTiny", '6':"RectTiny", '7':"RectTiny", '8':"RectSmall",
               '9':"RectSmall",'10':"RectMedium",'11':"RectMedium",
               '12':"RectBig",'13':"RectBig"}




# Generic list merging functions
def mergeOrNot(list1,list2):
    merge=False
    for item in list1:
        if item in list2:
            merge=True
            break
    return merge

def uniqueList(list1,list2):
    result = list1
    for j in list2:
        if j not in list1:
            result.append(j)
    return result

def cleverMergeLists(lists):
    anotherLoopRequired=False
    newList = []
    for myList in lists:
        addMyList=True
        if not anotherLoopRequired:
            for myList2 in lists:
                if not anotherLoopRequired:
                    if(myList==myList2):
                        continue
                    if(mergeOrNot(myList,myList2)):
                        anotherLoopRequired=True
                        addMyList=False
                        newList.append(uniqueList(myList,myList2))
                else:
                    newList.append(myList2)
            if(addMyList):
                newList.append(myList)
    if anotherLoopRequired:
        return cleverMergeLists(newList)
    else:
        return newList




# COMPUTER VISION ANALYSIS FUNCTIONS

# returns the MBRs for a given image
def boxes(orig):
    img = ImageOps.grayscale(orig)
    im = numpy.array(img)

    # Inner morphological gradient.
    im = morphology.grey_dilation(im, (3, 3)) - im

    # Binarize.
    mean, std = im.mean(), im.std()
    t = mean + std
    im[im < t] = 0
    im[im >= t] = 1

    # Connected components.
    lbl, numcc = label(im)
    # Size threshold.
    min_size = computer_vision_size_threshold # pixels
    box = []
    for i in range(1, numcc + 1):
        py, px = numpy.nonzero(lbl == i)
        if len(py) < min_size:
            im[lbl == i] = 0
            continue

        xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()
        # Four corners and centroid.
        box.append([
            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
            (numpy.mean(px), numpy.mean(py))])

    return im.astype(numpy.uint8) * 255, box 




# returns both the MBRs for a given image and the segmented sections of the image
def boxes_sep(orig):
    img = ImageOps.grayscale(orig)
    im = numpy.array(img)

    # Inner morphological gradient.
    im = morphology.grey_dilation(im, (3, 3)) - im

    # Binarize.
    mean, std = im.mean(), im.std()
    t = mean + std
    im[im < t] = 0
    im[im >= t] = 1

    # Connected components.
    lbl, numcc = label(im)
    
    # Size threshold.
    min_size = computer_vision_size_threshold # pixels
    box = []

    segmented_images = []
    for i in range(1, numcc + 1):
        im2 = deepcopy(lbl)
        py, px = numpy.nonzero(lbl == i)
        if len(py) < min_size:
            im[lbl == i] = 0
            continue
        segmented_images.append([])
        for j in range(len(lbl)):
            for k in range(len(lbl[j])):
                if lbl[j][k] == i:
                    segmented_images[-1].append([k,j])

    for i in range(1, numcc + 1):
        py, px = numpy.nonzero(lbl == i)
        if len(py) < min_size:
            im[lbl == i] = 0
            continue

        xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()
        # Four corners and centroid.
        box.append([
            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
            (numpy.mean(px), numpy.mean(py))])

    return im.astype(numpy.uint8) * 255, box, segmented_images




print("DOING COMPUTER VISION")

# find the corners for the given image
img = cv2.imread(sys.argv[1])
img_orig = copy.copy(img)
grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
corners = cv2.goodFeaturesToTrack(grayimg,100,corner_detection_quality_threshold,corner_detection_min_distance)
corners = numpy.float32(corners)
 
for item in corners:
  x,y = item[0]
  cv2.circle(img,(x,y),5,255,-1)
 
Image.fromarray(img).save("sketch_corners.jpg")

new_corners = []
for item in corners:
    x,y = item[0]
    new_corners.append([x,y])
corners = deepcopy(new_corners)

print("Number of corners found:")
print(len(corners))




# find the MBRs for the given image
orig = Image.open(sys.argv[1])
if (seperate_vision_corners == 1) and (corner_splitting_allowed == 1):
    im, box, seg_points = boxes_sep(orig)
else:
    im, box = boxes(orig)
Image.fromarray(im).save("sketch_lines.jpg")




# Draw perfect rectangles and the component centroid.
img = Image.fromarray(im)
visual = img.convert('RGB')
draw = ImageDraw.Draw(visual)
for b, centroid in box:
    draw.line(b + [b[0]], fill='yellow')
    cx, cy = centroid
    draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill='red')
visual.save("sketch_MBRs.jpg")




# Draw perfect rectangles and the component centroid.
img = Image.fromarray(im)
visual = img.convert('RGB')
draw = ImageDraw.Draw(visual)
for b, centroid in box:
    draw.rectangle([b[0],b[2]], fill='white')
visual.save("sketch_MBRs_filled.jpg")




all_boxes = []
# bounding boxes for all rectangles found
# all boxes is a list of all blocks [x,y,w,h], center points (x,y) and width (w) and height (h)
for b, centroid in box:
    width = float(b[1][0] - b[0][0])
    height = float(b[2][1] - b[0][1])
    center_x = float(b[0][0]+(width/2.0))
    center_y = float(b[0][1]+(height/2.0))
    all_boxes.append([center_x,center_y,width,height])
    #all_boxes.append([centroid[0],centroid[1],width,height])




# remove all boxes with a width or height less than size threshold (wrong)
# already done in computer vision section
new_all_boxes = []
for box in all_boxes:
    if (box[2] > threshold_block_width) and (box[3] > threshold_block_height):
        new_all_boxes.append(box)
all_boxes = deepcopy(new_all_boxes)




# remove all boxes that are fully inside other boxes (holes)
to_remove = []
for i in range(len(all_boxes)):
    for j in range(len(all_boxes)):
        if i!=j:
            if ((all_boxes[i][0]-(all_boxes[i][2]/2.0)) > (all_boxes[j][0]-(all_boxes[j][2]/2.0))):
                if ((all_boxes[i][0]+(all_boxes[i][2]/2.0)) < (all_boxes[j][0]+(all_boxes[j][2]/2.0))):
                    if ((all_boxes[i][1]-(all_boxes[i][3]/2.0)) > (all_boxes[j][1]-(all_boxes[j][3]/2.0))):
                        if ((all_boxes[i][1]+(all_boxes[i][3]/2.0)) < (all_boxes[j][1]+(all_boxes[j][3]/2.0))):
                            to_remove.append(i)

new_all_boxes = []
for i in range(len(all_boxes)):
    if i not in to_remove:
        new_all_boxes.append(all_boxes[i])
all_boxes = deepcopy(new_all_boxes)

if (seperate_vision_corners == 1) and (corner_splitting_allowed == 1):
    new_seg_points = []
    for i in range(len(seg_points)):
        if i not in to_remove:
            new_seg_points.append(seg_points[i])
    seg_points = deepcopy(new_seg_points)




# split non-rectangular orthogonal polygons into rectangles
if (corner_splitting_allowed==1):

    if seperate_vision_corners == 1:
        print("SPLITTING CORNERS")
        # associte each corner with the MBR whose original shape it is closest too
        corner_association = []
        for j in range(len(seg_points)):
            corner_association.append([])
        closest_corners = []
        to_remove = []
        for c in corners:
            min_distance = 99999999
            closest_seg = -1
            for j in range(len(seg_points)):
                for k in seg_points[j]:
                    x1 = c[0]
                    x2 = k[0]
                    y1 = c[1]
                    y2 = k[1]
                    distance = (sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
                    if distance < min_distance:
                        min_distance = distance
                        closest_seg = j
            if min_distance > max_distance_allowed:
                to_remove.append(c)
            else:
                closest_corners.append(closest_seg)
        for c in to_remove:
            corners.remove(c)
        for j in range(len(corners)):
            corner_association[closest_corners[j]].append(corners[j])

    else:
        # associate each corner with the MBR that it is within (problem if within two or more MBRs)
        corner_association = []
        for i in range(len(all_boxes)):
            corner_association.append([])
        for i in range(len(corners)):
            mbr_within = -1
            extra_give = 5
            found_counter = 0
            for j in range(len(all_boxes)):
                if corners[i][0] < all_boxes[j][0]+(all_boxes[j][2]/2.0)+extra_give:
                    if corners[i][0] > all_boxes[j][0]-(all_boxes[j][2]/2.0)-extra_give:
                        if corners[i][1] < all_boxes[j][1]+(all_boxes[j][3]/2.0)+extra_give:
                            if corners[i][1] > all_boxes[j][1]-(all_boxes[j][3]/2.0)-extra_give:
                                mbr_within = j
                                found_counter = found_counter+1
            if mbr_within == -1:
                print("error: no MBR found to associate with")
            if found_counter > 1:
                print("error: too many MBR possibilities")
            corner_association[mbr_within].append(corners[i])

    

    # split up every shape with more than 5 corners into multiple rectangles    
    final_to_remove = []
    final_to_add = []
    for i in range(len(all_boxes)):
        if len(corner_association[i]) > 5:

            if (len(corner_association[i]) % 2) == 1:
                print("error: odd number of associated corners")

            # make the y values similar
            split_lines_y = []
            split_y = []
            for c in corner_association[i]:
                found = 0
                for y in range(len(split_lines_y)):
                    max_y = max([sublist[1] for sublist in split_lines_y[y]])
                    min_y = min([sublist[1] for sublist in split_lines_y[y]])
                    if min_y < c[1] + threshold_corner_amount_y:
                        if max_y > c[1] - threshold_corner_amount_y:
                            split_lines_y[y].append(c)
                            found = found+1
                            
                if found == 0:
                    split_lines_y.append([c])
                    split_y.append([])
                if found > 1:
                    print("error: multiple y values found")
            
            if add_togethor_similar_y == 1:
                split_lines_y = cleverMergeLists(split_lines_y)

            for y in range(len(split_lines_y)):
                split_y[y] = 0 
                for j in split_lines_y[y]:
                    split_y[y] = split_y[y] + j[1]
                split_y[y] = split_y[y] / len(split_lines_y[y])

            new_cor = []
            for c in range(len(corner_association[i])):
                match = 0
                for j in range(len(split_lines_y)):
                    if corner_association[i][c] in split_lines_y[j]:  
                        match = 1  
                        new_cor.append([corner_association[i][c][0],split_y[j]])
                if match == 0:
                    print("error: no matching y value found")
            corner_association[i] = deepcopy(new_cor)



            # make the x values similar
            split_lines_x = []
            split_x = []
            for c in corner_association[i]:
                found = 0
                for x in range(len(split_lines_x)):
                    max_x = max([sublist[0] for sublist in split_lines_x[x]])
                    min_x = min([sublist[0] for sublist in split_lines_x[x]])
                    if min_x < c[0] + threshold_corner_amount_x:
                        if max_x > c[0] - threshold_corner_amount_x:
                            split_lines_x[x].append(c)
                            found = found+1
                            
                if found == 0:
                    split_lines_x.append([c])
                    split_x.append([])
                if found > 1:
                    print("error: multiple x values found")
            
            if add_togethor_similar_x == 1:
                split_lines_x = cleverMergeLists(split_lines_x)

            for x in range(len(split_lines_x)):
                split_x[x] = 0 
                for j in split_lines_x[x]:
                    split_x[x] = split_x[x] + j[0]
                split_x[x] = split_x[x] / len(split_lines_x[x])

            new_cor = []
            for c in range(len(corner_association[i])):
                match = 0
                for j in range(len(split_lines_x)):
                    if corner_association[i][c] in split_lines_x[j]:  
                        match = 1  
                        new_cor.append([split_x[j],corner_association[i][c][1]])
                if match == 0:
                    print("error: no matching x value found")
            corner_association[i] = deepcopy(new_cor)



            # find horizontal and vertical edges
            y_orderings = []
            x_orderings = []
            edges_all_x = []
            edges_all_y = []
            for c in corner_association[i]:
                chosen_x = 0
                chosen_y = 0
                for j in range(len(x_orderings)):
                    if c[0] == x_orderings[j][0][0]:
                        x_orderings[j].append(c)
                        chosen_x = 1
                if chosen_x == 0:
                    x_orderings.append([c])
                for j in range(len(y_orderings)):
                    if c[1] == y_orderings[j][0][1]:
                        y_orderings[j].append(c)
                        chosen_y = 1
                if chosen_y == 0:
                    y_orderings.append([c])
            for xx in range(len(x_orderings)):
                x_orderings[xx] = sorted(x_orderings[xx], key = lambda x: int(x[1]))
            for yy in range(len(y_orderings)):
                y_orderings[yy] = sorted(y_orderings[yy], key = lambda x: int(x[0]))

            connect = True
            for o in range(len(x_orderings)):
                for c in range(len(x_orderings[o])):
                    if c < len(x_orderings[o]):
                        if connect == True:
                            edges_all_x.append([x_orderings[o][c],x_orderings[o][c+1]])
                            connect = False
                        else:
                            connect = True

            for o in range(len(y_orderings)):
                for c in range(len(y_orderings[o])):
                    if c < len(y_orderings[o]):
                        if connect == True:
                            edges_all_y.append([y_orderings[o][c],y_orderings[o][c+1]])
                            connect = False
                        else:
                            connect = True



            # associate edges with each corner
            corner_edges = []
            for c in corner_association[i]:
                edge_ver = []
                edge_hor = []
                for e in edges_all_x:
                    if c in e:
                        edge_ver = e
                for e in edges_all_y:
                    if c in e:
                        edge_hor = e
                corner_edges.append([edge_hor,edge_ver])



            # identify concave and convex corners
            convex_corners = []         # point outside
            concave_corners = []        # point inside (ones that we want/use)
            ori_edges_all_x = deepcopy(edges_all_x)
            for j in range(len(corner_edges)):
                point_to_test = deepcopy(corner_association[i][j])
                shift_amount_corner_test = 0.01 

                if corner_edges[j][0][0][0] < corner_edges[j][0][1][0]:
                    if corner_edges[j][0][0][0] == point_to_test[0]:
                        point_to_test[0] = point_to_test[0]-shift_amount_corner_test
                    else:
                        point_to_test[0] = point_to_test[0]+shift_amount_corner_test
                else:
                    if corner_edges[j][0][0][0] == point_to_test[0]:
                        point_to_test[0] = point_to_test[0]+shift_amount_corner_test
                    else:
                        point_to_test[0] = point_to_test[0]-shift_amount_corner_test

                if corner_edges[j][1][0][1] < corner_edges[j][1][1][1]:
                    if corner_edges[j][1][0][1] == point_to_test[1]:
                        point_to_test[1] = point_to_test[1]-shift_amount_corner_test
                    else:
                        point_to_test[1] = point_to_test[1]+shift_amount_corner_test
                else:
                    if corner_edges[j][1][0][1] == point_to_test[1]:
                        point_to_test[1] = point_to_test[1]+shift_amount_corner_test
                    else:
                        point_to_test[1] = point_to_test[1]-shift_amount_corner_test
                    
                num_line_intersections = 0
                for linex in edges_all_x:
                    if linex[0][1] < linex[1][1]:
                        if point_to_test[1] < linex[1][1]:
                            if point_to_test[1] > linex[0][1]:
                                if point_to_test[0] > linex[0][0]:
                                    num_line_intersections = num_line_intersections + 1
                    else:
                        if point_to_test[1] > linex[1][1]:
                            if point_to_test[1] < linex[0][1]:
                                if point_to_test[0] > linex[0][0]:
                                    num_line_intersections = num_line_intersections + 1

                if (num_line_intersections%2) == 0:
                    convex_corners.append(j)
                else:
                    concave_corners.append(j)



            # identify extra horzontal edges between concave corners
            extra_edges_hor = []     
            for j in concave_corners:
                current_point = corner_association[i][j]
                intersecting_lines = []
                for linex in edges_all_x:
                    if linex[0][0]!=current_point[0]:
                        if linex[0][1] < linex[1][1]:
                            if current_point[1] < linex[1][1]+shift_amount_corner_test: 
                                if current_point[1] > linex[0][1]-shift_amount_corner_test: 
                                    intersecting_lines.append(linex)
                        else:
                            if current_point[1] > linex[1][1]-shift_amount_corner_test: 
                                if current_point[1] < linex[0][1]+shift_amount_corner_test: 
                                    intersecting_lines.append(linex)

                left_intersecting_closest = []
                left_distance = 99999999
                right_intersecting_closest = []
                right_distance = 99999999
                for line in intersecting_lines:
                    if current_point[0] > line[0][0]:
                        if current_point[0] - line[0][0] < left_distance:
                            left_distance = current_point[0] - line[0][0] 
                            left_intersecting_closest = line
                    else:
                        if line[0][0] - current_point[0] < right_distance:
                            right_distance = line[0][0] - current_point[0]
                            right_intersecting_closest = line

                extra_edges_hor.append([current_point,[right_intersecting_closest[0][0],current_point[1]]])
                extra_edges_hor.append([[left_intersecting_closest[0][0],current_point[1]],current_point])



            # identify extra vertical edges between concave corners
            extra_edges_ver = []     
            for j in concave_corners:
                current_point = corner_association[i][j]
                intersecting_lines = []
                for liney in edges_all_y:
                    if liney[0][1]!=current_point[1]:
                        if liney[0][0] < liney[1][0]:
                            if current_point[0] < liney[1][0]+shift_amount_corner_test: 
                                if current_point[0] > liney[0][0]-shift_amount_corner_test: 
                                    intersecting_lines.append(liney)
                        else:
                            if current_point[0] > liney[1][0]-shift_amount_corner_test: 
                                if current_point[0] < liney[0][0]+shift_amount_corner_test: 
                                    intersecting_lines.append(liney)

                up_intersecting_closest = []
                up_distance = 99999999
                down_intersecting_closest = []
                down_distance = 99999999
                for line in intersecting_lines:
                    if current_point[1] > line[0][1]:
                        if current_point[1] - line[0][1] < up_distance:
                            up_distance = current_point[1] - line[0][1] 
                            up_intersecting_closest = line
                    else:
                        if line[0][1] - current_point[1] < down_distance:
                            down_distance = line[0][1] - current_point[1]
                            down_intersecting_closest = line

                extra_edges_ver.append([current_point,[current_point[0],up_intersecting_closest[0][1]]])
                extra_edges_ver.append([[current_point[0],down_intersecting_closest[0][1]],current_point])



            # remove duplicates
            extra_edges_ver2 = []
            extra_edges_hor2 = []
            for j in extra_edges_ver:
                if j not in extra_edges_ver2:
                    extra_edges_ver2.append(j)
            for j in extra_edges_hor:
                if j not in extra_edges_hor2:
                    extra_edges_hor2.append(j)
            extra_edges_ver = deepcopy(extra_edges_ver2)
            extra_edges_hor = deepcopy(extra_edges_hor2)



            #order edges (left to right, top to bottom)       
            for edge_test in range(len(extra_edges_ver)):
                if extra_edges_ver[edge_test][0][1] > extra_edges_ver[edge_test][1][1]:
                    extra_edges_ver[edge_test] = [extra_edges_ver[edge_test][1],extra_edges_ver[edge_test][0]]

            for edge_test in range(len(extra_edges_hor)):
                if extra_edges_hor[edge_test][0][0] > extra_edges_hor[edge_test][1][0]:
                    extra_edges_hor[edge_test] = [extra_edges_hor[edge_test][1],extra_edges_hor[edge_test][0]]


            for edge_test in range(len(edges_all_x)):
                if edges_all_x[edge_test][0][1] > edges_all_x[edge_test][1][1]:
                    edges_all_x[edge_test] = [edges_all_x[edge_test][1],edges_all_x[edge_test][0]]

            for edge_test in range(len(edges_all_y)):
                if edges_all_y[edge_test][0][0] > edges_all_y[edge_test][1][0]:
                    edges_all_y[edge_test] = [edges_all_y[edge_test][1],edges_all_y[edge_test][0]]



            #split extra edges into two if it intersects another extra edge 
            no_change = 0
            while(no_change==0):
                to_add_hor = []
                to_add_ver = []
                to_remove_hor = []
                to_remove_ver = []
                no_change = 1
                for j in extra_edges_hor:
                    for k in extra_edges_ver:
                        if j[0][0] < k[0][0]:
                            if j[1][0] > k[0][0]:
                                if k[0][1] < j[0][1]:
                                    if k[1][1] > j[0][1]:
                                        to_add_hor.append([j[0],[k[0][0],j[0][1]]])
                                        to_add_hor.append([[k[0][0],j[0][1]],j[1]])
                                        to_remove_hor.append(j)
                                        to_add_ver.append([k[0],[k[0][0],j[0][1]]])
                                        to_add_ver.append([[k[0][0],j[0][1]],k[1]])
                                        to_remove_ver.append(k)
                                        no_change = 0
                if no_change == 0:
                    extra_edges_hor.append(to_add_hor[0])
                    extra_edges_hor.append(to_add_hor[1])
                    extra_edges_hor.remove(to_remove_hor[0])
                    extra_edges_ver.append(to_add_ver[0])
                    extra_edges_ver.append(to_add_ver[1])                    
                    extra_edges_ver.remove(to_remove_ver[0])



            #get all touching line points for creating small blocks
            all_touching_line_points = []
            for j in corner_association[i]:
                if j not in all_touching_line_points:
                    all_touching_line_points.append(j)
            for j in extra_edges_ver:
                for k in j:
                    if k not in all_touching_line_points:
                        all_touching_line_points.append(k)
            for j in extra_edges_hor:
                for k in j:
                    if k not in all_touching_line_points:
                        all_touching_line_points.append(k)
            


            # mark extra points that were not already corners
            extra_added_points = []
            for j in all_touching_line_points:
                if j not in corner_association[i]:
                    extra_added_points.append(j)



            #order edges (left to right, top to bottom)           
            for edge_test in range(len(extra_edges_ver)):
                if extra_edges_ver[edge_test][0][1] > extra_edges_ver[edge_test][1][1]:
                    extra_edges_ver[edge_test] = [extra_edges_ver[edge_test][1],extra_edges_ver[edge_test][0]]

            for edge_test in range(len(extra_edges_hor)):
                if extra_edges_hor[edge_test][0][0] > extra_edges_hor[edge_test][1][0]:
                    extra_edges_hor[edge_test] = [extra_edges_hor[edge_test][1],extra_edges_hor[edge_test][0]]


            for edge_test in range(len(edges_all_x)):
                if edges_all_x[edge_test][0][1] > edges_all_x[edge_test][1][1]:
                    edges_all_x[edge_test] = [edges_all_x[edge_test][1],edges_all_x[edge_test][0]]

            for edge_test in range(len(edges_all_y)):
                if edges_all_y[edge_test][0][0] > edges_all_y[edge_test][1][0]:
                    edges_all_y[edge_test] = [edges_all_y[edge_test][1],edges_all_y[edge_test][0]]



            #split lines into sub-lines based on extra contact edges added
            no_change_split = 0
            while(no_change_split == 0):
                no_change_split = 1
                to_remove = []
                to_add = []
                for j in edges_all_x:
                    for k in extra_added_points:
                        if k[1] < j[1][1]:
                            if k[1] > j[0][1]:
                                if k[0] == j[0][0]:
                                    to_remove.append(j)
                                    to_add.append([j[0],k])
                                    to_add.append([k,j[1]])
                                    no_change_split = 0

                if no_change_split == 0:
                    edges_all_x.remove(to_remove[0])
                    edges_all_x.append(to_add[0])
                    edges_all_x.append(to_add[1])

                else:
                    for j in edges_all_y:
                        for k in extra_added_points:
                            if k[0] < j[1][0]:
                                if k[0] > j[0][0]:
                                    if k[1] == j[0][1]:
                                        to_remove.append(j)
                                        to_add.append([j[0],k])
                                        to_add.append([k,j[1]])
                                        no_change_split = 0

                    if no_change_split == 0:
                        edges_all_y.remove(to_remove[0])
                        edges_all_y.append(to_add[0])
                        edges_all_y.append(to_add[1])



            # remove duplicates and order
            for new_edge_x in extra_edges_ver:
                if new_edge_x not in edges_all_x:
                    edges_all_x.append(new_edge_x)
            for new_edge_y in extra_edges_hor:
                if new_edge_y not in edges_all_y:
                    edges_all_y.append(new_edge_y)

            small_edges_hor = deepcopy(edges_all_y)
            small_edges_ver = deepcopy(edges_all_x)

            for edge_test in range(len(small_edges_ver)):
                if small_edges_ver[edge_test][0][1] > small_edges_ver[edge_test][1][1]:
                    small_edges_ver[edge_test] = [small_edges_ver[edge_test][1],small_edges_ver[edge_test][0]]

            for edge_test in range(len(small_edges_hor)):
                if small_edges_hor[edge_test][0][0] > small_edges_hor[edge_test][1][0]:
                    small_edges_hor[edge_test] = [small_edges_hor[edge_test][1],small_edges_hor[edge_test][0]]
            


            #get all the small boxes (maximum)
            new_boxes = []
            for j in small_edges_hor:
                for k in small_edges_hor:
                    above = 0
                    below = 0
                    connect_left = []
                    connect_right = []
                    if j!=k:
                        if k[0][0] == j[0][0]:
                            if k[1][0] == j[1][0]:
                                if k[0][1] > j[0][1]:
                                    below = 1
                                else:
                                    above = 1
                    if below == 1:
                        for m in small_edges_ver:
                            if m[0] == j[0]:
                                if m[1] == k[0]:
                                    connect_left = m
                            if m[0] == j[1]:
                                if m[1] == k[1]:
                                    connect_right = m
                    if above == 1:
                        for m in small_edges_ver:
                            if m[0] == k[0]:
                                if m[1] == j[0]:
                                    conect_left = m
                            if m[0] == k[1]:
                                if m[1] == j[1]:
                                    connect_right = m

                    if (above == 1) and (connect_left != []) and (connect_right != []):
                        new_boxes.append([k,connect_right,j,connect_left])
                    if (below == 1) and (connect_left != []) and (connect_right != []):
                        new_boxes.append([j,connect_right,k,connect_left])




            #convert to correct format
            new_boxes2 = []
            for j in new_boxes:
                width = j[0][1][0] - j[0][0][0]
                height = j[1][1][1] - j[1][0][1]
                center_x = j[0][0][0] + (width/2.0)
                center_y = j[1][0][1] + (height/2.0)
                new_boxes2.append([center_x,center_y,width,height])


            

            # remove boxes that are actually holes
            new_boxes3 = []
            for j in new_boxes2:
                num_line_intersections = 0
                point_to_test = [j[0],j[1]]
                for linex in ori_edges_all_x:
                    if linex[0][1] < linex[1][1]:
                        if point_to_test[1] < linex[1][1]:
                            if point_to_test[1] > linex[0][1]:
                                if point_to_test[0] > linex[0][0]:
                                    num_line_intersections = num_line_intersections + 1
                    else:
                        if point_to_test[1] > linex[1][1]:
                            if point_to_test[1] < linex[0][1]:
                                if point_to_test[0] > linex[0][0]:
                                    num_line_intersections = num_line_intersections + 1

                if (num_line_intersections%2) == 1:
                    new_boxes3.append(j)



            # merge two boxes togethor if they are horizontally next to each other and have the same height
            new_boxes4 = deepcopy(new_boxes3)
            no_change = 1
            to_merge = [0]
            while(len(to_merge)>0):
                to_merge = []
                no_change = 0
                for j in new_boxes4:
                    for k in new_boxes4:
                        if j != k:
                            if abs(j[1] - k[1]) < 0.1:             
                                if abs(j[3] - k[3]) < 0.1:         
                                    if abs((j[0]+(j[2]/2.0)) - (k[0]-(k[2]/2.0))) < 0.1:
                                        to_merge.append([j,k])
                if len(to_merge)>0:
                    j = to_merge[0][0]
                    k = to_merge[0][1]
                    width = j[2]+k[2]
                    height = j[3]
                    center_x = (j[0]-(j[2]/2.0)) + (width/2.0)
                    center_y = j[1]
                    new_boxes4.append([center_x,center_y,width,height])
                    new_boxes4.remove(j)
                    new_boxes4.remove(k)



            # add the new boxes to all_boxes and remove the original
            final_to_remove.append(all_boxes[i])
            for j in new_boxes4:
                final_to_add.append(j)

    for i in final_to_remove:
        all_boxes.remove(i)
    for i in final_to_add:
        all_boxes.append(i)




stab_all_boxes = deepcopy(all_boxes)
for i in range(len(stab_all_boxes)):
    stab_all_boxes[i][1] = (-1*(stab_all_boxes[i][1]))+2000

lowest_y = 99999999
for i in stab_all_boxes:
    if i[1]-(i[3]/2.0) < lowest_y:
        lowest_y = i[1]-(i[3]/2.0)
down_amount = lowest_y - 100.0

for i in range(len(stab_all_boxes)):
    stab_all_boxes[i][1] = stab_all_boxes[i][1] - down_amount

f = open("sketch_blocks_data.txt", "w")
for i in stab_all_boxes:
    f.write('%s %s %s %s\n' % (i[0],i[1]-(i[3]/2.0),i[2],i[3]))
f.close()




#find the largest and smallest block dimensions:
largest_value = 0
smallest_value = 99999999
largest_width = 0
smallest_width = 99999999
largest_height = 0
smallest_height = 99999999
widths = []
heights = []
areas = []
center_mass_ori_x = 0
center_mass_ori_y = 0
total_mass_ori = 0

for box in all_boxes:
    widths.append(box[2])
    heights.append(box[3])
    areas.append(box[2]*box[3])
    center_mass_ori_x = center_mass_ori_x + (box[0]*box[2]*box[3])
    center_mass_ori_y = center_mass_ori_y + (box[1]*box[2]*box[3])
    total_mass_ori = total_mass_ori + (box[2]*box[3])

    if box[2] > largest_value:
        largest_value = box[2]
    if box[2] < smallest_value:
        smallest_value = box[2]
    if box[3] > largest_value:
        largest_value = box[3]
    if box[3] < smallest_value:
        smallest_value = box[3]

    if box[2] > largest_width:
        largest_width = box[2]
    if box[2] < smallest_width:
        smallest_width = box[2]

    if box[3] > largest_height:
        largest_height = box[3]
    if box[3] < smallest_height:
        smallest_height = box[3]

center_mass_ori_x = center_mass_ori_x / total_mass_ori
center_mass_ori_y = center_mass_ori_y / total_mass_ori

sizes = widths+heights
mean_width = mean(widths)
mean_height = mean(heights)
mean_size = mean(sizes)
mean_area = mean(areas)
median_width = median(widths)
median_height = median(heights)
median_size = median(sizes)
median_area = median(areas)

actual_block_sizes = []
for key,value in blocks.items():
    actual_block_sizes.append(value[0])
actual_block_mean = mean(actual_block_sizes)
actual_block_median = median(actual_block_sizes)




maximum_width_gap_touching = 0                              # extra number of pixels to add to a blocks width when determining touching blocks 
maximum_height_gap_touching = smallest_value*3 - 1          # extra number of pixels to add to a blocks height when determining touching blocks




# finds all supporters (direct and indirect) for a given block
def get_all_supporters(query):
    indirect = []
    to_check = [query]
    while len(to_check) > 0:
        doin = to_check.pop()
        indirect.append(doin)
        if doin != 9999:
            for j in graph_supporters[doin]:
                if j not in indirect:
                    if j not in to_check:
                        to_check.append(j)
    new_indirect = []
    for i in indirect:
      if i not in new_indirect:
        new_indirect.append(i)
    new_indirect.remove(query)
    return new_indirect




# finds all supportees (direct and indirect) for a given block
def get_all_supportees(query):
    indirect = []
    to_check = [query]
    while len(to_check) > 0:
        doin = to_check.pop()
        indirect.append(doin)
        for j in graph_supportees[doin]:
            if j not in indirect:
                if j not in to_check:
                    to_check.append(j)
    new_indirect = []
    for i in indirect:
      if i not in new_indirect:
        new_indirect.append(i)
    new_indirect.remove(query)
    return new_indirect




# finds all support paths from start block, upwards to end block
def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths




# not used but always good to have in case
def find_shortest_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if start not in graph:
        return None
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest




# check if structure has local stability
def check_local_stability(all_boxes):
    for i in range(len(all_boxes)):
        left_support = 0
        right_support = 0
        box = all_boxes[i]
        for j in graph_supporters[i]:
            if j == 9999:
                left_support = 1
                right_support = 1
            else:
                box2 = all_boxes[j]
                box2_left = box2[0]-(box2[2]/2.0)
                box2_right = box2[0]+(box2[2]/2.0)
                if box2_left < box[0]:
                    left_support = 1
                if box2_right > box[0]:
                    right_support = 1
        if left_support == 0:
            print("UNSTABLE LOCAL (L) !!!!!")
            print(i)
        if right_support == 0:
            print("UNSTABLE LOCAL (R) !!!!!")
            print(i)  




def isCentreSupporter(RAx):
    if (RAx =="ERA.MOST_START_I" or RAx =="ERA.LESS_START_I" or RAx =="ERA.MOST_FINISH_I" or RAx =="ERA.LESS_FINISH_I" or RAx =="ERA.CENTRE_DURING" or RAx =="ERA.CENTRE_DURING_I" or RAx =="ERA.LEFT_DURING_I" or RAx =="ERA.RIGHT_DURING_I" or RAx =="ERA.MOST_START" or RAx =="ERA.MOST_FINISH" or RAx =="ERA.MOST_OVERLAP_MOST" or RAx =="ERA.LESS_OVERLAP_MOST" or RAx =="ERA.MOST_OVERLAP_MOST_I" or RAx =="ERA.MOST_OVERLAP_LESS_I" or RAx =="ERA.EQUAL"):
        return True
    return False

def isLeftSupporter(RAx):
    if (RAx =="ERA.LESS_OVERLAP_LESS" or RAx =="ERA.MOST_OVERLAP_LESS" or RAx =="ERA.LESS_START" or RAx =="ERA.LEFT_DURING"):
        return True
    return False

def isRightSupporter(RAx):
    if (RAx =="ERA.LESS_OVERLAP_MOST_I" or RAx =="ERA.LESS_OVERLAP_LESS_I" or RAx =="ERA.LESS_FINISH" or RAx =="ERA.RIGHT_DURING"):
        return True
    return False




# Calculate the ERA relations based on touching blocks
def find_era_relation(touching_line):
    ERA_relations = []
    ERA_threshold = 0.06

    s1 = touching_line[0]
    s2 = touching_line[2]       # these are in the wrong order (should be 2,0,3,1) but the RAx rules are also wrong (filpped)
    e1 = touching_line[1]
    e2 = touching_line[3]
    RA = "unknown"
    if (s2 - e1 >=ERA_threshold):
        RA ="ERA.BEFORE"
    elif (s1 - e2 >=ERA_threshold):
        RA ="ERA.AFTER"
    elif (s2 - e1 <ERA_threshold and s2 - e1 >= 0 and s1 < e2):
        RA ="ERA.MEET"
    elif (s1 - e2 <ERA_threshold and s1 - e2 >= 0 and s2 < e1):
        RA ="ERA.MEET_I"
    elif (s1 == s2 and e2 - e1 >= 0 and (e2 - s2) / 2 < e1 - s1):
        RA ="ERA.MOST_START"
    elif (s1 == s2 and e1 - e2 > 0 and e2 - s2 > (e1 - s1) / 2):
        RA ="ERA.MOST_START_I"
    elif (s1 == s2 and e2 - e1 > 0 and (e2 - s2) / 2 >= e1 - s1):
        RA ="ERA.LESS_START"
    elif (s1 == s2 and e1 - e2 > 0 and e2 - s2 <= (e1 - s1) / 2):
        RA ="ERA.LESS_START_I"
    elif (s1 - s2 > 0 and e2 - e1 > 0 and e1 <= (s2 + e2) / 2):
        RA ="ERA.LEFT_DURING"
    elif (s2 - s1 > 0 and e1 - e2 > 0 and e2 <= (s1 + e1) / 2):
        RA ="ERA.LEFT_DURING_I"
    elif (s1 - s2 > 0 and e2 - e1 > 0 and s1 >= (s2 + e2) / 2):
        RA ="ERA.RIGHT_DURING"
    elif (s2 - s1 > 0 and e1 - e2 > 0 and s2 >= (s1 + e1) / 2):
        RA ="ERA.RIGHT_DURING_I"
    elif (s1 - s2 > 0 and e2 - e1 > 0 and s1 < (s2 + e2) / 2 and e1 > (s2 + e2) / 2):
        RA ="ERA.CENTRE_DURING"
    elif (s2 - s1 > 0 and e1 - e2 > 0 and s2 < (s1 + e1) / 2 and e2 > (s1 + e1) / 2):
        RA ="ERA.CENTRE_DURING_I"
    elif (s1 - s2 > 0 and e1 == e2 and (e2 - s2) / 2 < e1 - s1):
        RA ="ERA.MOST_FINISH"
    elif (s2 - s1 > 0 and e1 == e2 and e2 - s2 > (e1 - s1) / 2):
        RA ="ERA.MOST_FINISH_I"
    elif (s1 - s2 > 0 and e1 == e2 and (e2 - s2) / 2 >= e1 - s1):
        RA ="ERA.LESS_FINISH"
    elif (s2 - s1 > 0 and e1 == e2 and e2 - s2 <= (e1 - s1) / 2):
        RA ="ERA.LESS_FINISH_I"
    elif (abs(s1 - s2) <ERA_threshold and abs(e1 - e2) <ERA_threshold):
        RA ="ERA.EQUAL"
    elif (s2 - s1 > 0 and e2 - e1 > 0 and e1 - s2 > 0 and e1 - s2 >= s2 - s1 and e1 - s2 >= e2 - e1):
        RA ="ERA.MOST_OVERLAP_MOST"
    elif (s2 - s1 > 0 and e2 - e1 > 0 and e1 - s2 > 0 and e1 - s2 < s2 - s1 and e1 - s2 >= e2 - e1):
        RA ="ERA.LESS_OVERLAP_MOST"
    elif (s2 - s1 > 0 and e2 - e1 > 0 and e1 - s2 > 0 and e1 - s2 >= s2 - s1 and e1 - s2 < e2 - e1):
        RA ="ERA.MOST_OVERLAP_LESS"
    elif (s2 - s1 > 0 and e2 - e1 > 0 and e1 - s2 > 0 and e1 - s2 < s2 - s1 and e1 - s2 < e2 - e1):
        RA ="ERA.LESS_OVERLAP_LESS"
    elif (s1 - s2 > 0 and e1 - e2 > 0 and e1 - s2 > 0 and e2 - s1 >= s1 - s2 and e2 - s1 >= e1 - e2):
        RA ="ERA.MOST_OVERLAP_MOST_I"
    elif (s1 - s2 > 0 and e1 - e2 > 0 and e2 - s1 > 0 and e2 - s1 < s1 - s2 and e2 - s1 >= e1 - e2):
        RA ="ERA.LESS_OVERLAP_MOST_I"
    elif (s1 - s2 > 0 and e1 - e2 > 0 and e2 - s1 > 0 and e2 - s1 >= s1 - s2 and e2 - s1 < e1 - e2):
        RA ="ERA.MOST_OVERLAP_LESS_I"
    elif (s1 - s2 > 0 and e1 - e2 > 0 and e2 - s1 > 0 and e2 - s1 < s1 - s2 and e2 - s1 < e1 - e2):
        RA ="ERA.LESS_OVERLAP_LESS_I"
    ERA_relations.append(RA)
    return ERA_relations




# Calculate the ERA relations based on touching blocks
def find_era_relations():
    ERA_relations = []
    ERA_threshold = 5
    for i in range(len(touching_lines)):
        s1 = touching_lines[i][0]
        s2 = touching_lines[i][2]       # these are in the wrong order (should be 2,0,3,1) but the RAx rules are also wrong (filpped)
        e1 = touching_lines[i][1]
        e2 = touching_lines[i][3]
        RA = "unknown"
        if (s2 - e1 >=ERA_threshold):
            RA ="ERA.BEFORE"
        elif (s1 - e2 >=ERA_threshold):
            RA ="ERA.AFTER"
        elif (s2 - e1 <ERA_threshold and s2 - e1 >= 0 and s1 < e2):
            RA ="ERA.MEET"
        elif (s1 - e2 <ERA_threshold and s1 - e2 >= 0 and s2 < e1):
            RA ="ERA.MEET_I"
        elif (s1 == s2 and e2 - e1 >= 0 and (e2 - s2) / 2 < e1 - s1):
            RA ="ERA.MOST_START"
        elif (s1 == s2 and e1 - e2 > 0 and e2 - s2 > (e1 - s1) / 2):
            RA ="ERA.MOST_START_I"
        elif (s1 == s2 and e2 - e1 > 0 and (e2 - s2) / 2 >= e1 - s1):
            RA ="ERA.LESS_START"
        elif (s1 == s2 and e1 - e2 > 0 and e2 - s2 <= (e1 - s1) / 2):
            RA ="ERA.LESS_START_I"
        elif (s1 - s2 > 0 and e2 - e1 > 0 and e1 <= (s2 + e2) / 2):
            RA ="ERA.LEFT_DURING"
        elif (s2 - s1 > 0 and e1 - e2 > 0 and e2 <= (s1 + e1) / 2):
            RA ="ERA.LEFT_DURING_I"
        elif (s1 - s2 > 0 and e2 - e1 > 0 and s1 >= (s2 + e2) / 2):
            RA ="ERA.RIGHT_DURING"
        elif (s2 - s1 > 0 and e1 - e2 > 0 and s2 >= (s1 + e1) / 2):
            RA ="ERA.RIGHT_DURING_I"
        elif (s1 - s2 > 0 and e2 - e1 > 0 and s1 < (s2 + e2) / 2 and e1 > (s2 + e2) / 2):
            RA ="ERA.CENTRE_DURING"
        elif (s2 - s1 > 0 and e1 - e2 > 0 and s2 < (s1 + e1) / 2 and e2 > (s1 + e1) / 2):
            RA ="ERA.CENTRE_DURING_I"
        elif (s1 - s2 > 0 and e1 == e2 and (e2 - s2) / 2 < e1 - s1):
            RA ="ERA.MOST_FINISH"
        elif (s2 - s1 > 0 and e1 == e2 and e2 - s2 > (e1 - s1) / 2):
            RA ="ERA.MOST_FINISH_I"
        elif (s1 - s2 > 0 and e1 == e2 and (e2 - s2) / 2 >= e1 - s1):
            RA ="ERA.LESS_FINISH"
        elif (s2 - s1 > 0 and e1 == e2 and e2 - s2 <= (e1 - s1) / 2):
            RA ="ERA.LESS_FINISH_I"
        elif (abs(s1 - s2) <ERA_threshold and abs(e1 - e2) <ERA_threshold):
            RA ="ERA.EQUAL"
        elif (s2 - s1 > 0 and e2 - e1 > 0 and e1 - s2 > 0 and e1 - s2 >= s2 - s1 and e1 - s2 >= e2 - e1):
            RA ="ERA.MOST_OVERLAP_MOST"
        elif (s2 - s1 > 0 and e2 - e1 > 0 and e1 - s2 > 0 and e1 - s2 < s2 - s1 and e1 - s2 >= e2 - e1):
            RA ="ERA.LESS_OVERLAP_MOST"
        elif (s2 - s1 > 0 and e2 - e1 > 0 and e1 - s2 > 0 and e1 - s2 >= s2 - s1 and e1 - s2 < e2 - e1):
            RA ="ERA.MOST_OVERLAP_LESS"
        elif (s2 - s1 > 0 and e2 - e1 > 0 and e1 - s2 > 0 and e1 - s2 < s2 - s1 and e1 - s2 < e2 - e1):
            RA ="ERA.LESS_OVERLAP_LESS"
        elif (s1 - s2 > 0 and e1 - e2 > 0 and e1 - s2 > 0 and e2 - s1 >= s1 - s2 and e2 - s1 >= e1 - e2):
            RA ="ERA.MOST_OVERLAP_MOST_I"
        elif (s1 - s2 > 0 and e1 - e2 > 0 and e2 - s1 > 0 and e2 - s1 < s1 - s2 and e2 - s1 >= e1 - e2):
            RA ="ERA.LESS_OVERLAP_MOST_I"
        elif (s1 - s2 > 0 and e1 - e2 > 0 and e2 - s1 > 0 and e2 - s1 >= s1 - s2 and e2 - s1 < e1 - e2):
            RA ="ERA.MOST_OVERLAP_LESS_I"
        elif (s1 - s2 > 0 and e1 - e2 > 0 and e2 - s1 > 0 and e2 - s1 < s1 - s2 and e2 - s1 < e1 - e2):
            RA ="ERA.LESS_OVERLAP_LESS_I"
        ERA_relations.append(RA)
    return ERA_relations




# Use the ERA rules to determine is the sketch drawing is stable (qualitative LOCAL)
def calc_era_stability(all_boxes, ERA_relations,touching_blocks):
    ERA_stable = []
    no_change_ERA = 0

    for i in range(len(all_boxes)):
        ERA_stable.append(0)

    while(no_change_ERA == 0):
        old_ERA_stable = deepcopy(ERA_stable)

        for i in range(len(all_boxes)):
            rightSupporter = False
            leftSupporter = False
            centreSupporter = False

            if graph_supporters[i] == [9999]:
                ERA_stable[i] = 1

            else:
                for j in graph_supporters[i]:
                    if (ERA_stable[j]==1):  
                        for x in range(len(ERA_relations)):
                            if touching_blocks[x] == [j,i]:
                                if isLeftSupporter(ERA_relations[x]): 
                                    leftSupporter = True
                     
                for k in graph_supporters[i]:
                    if (ERA_stable[k]==1): 
                        for y in range(len(ERA_relations)):
                            if touching_blocks[y] == [k,i]:
                                if isRightSupporter(ERA_relations[y]):
                                    rightSupporter = True

                for m in graph_supporters[i]:
                    if (ERA_stable[m]==1):
                        for z in range(len(ERA_relations)):
                            if touching_blocks[z] == [m,i]:
                                if isCentreSupporter(ERA_relations[z]):
                                    centreSupporter = True    

            if ((leftSupporter and rightSupporter) or centreSupporter):       
                ERA_stable[i] = 1

        if (sorted(ERA_stable) == sorted(old_ERA_stable)):
            no_change_ERA = 1

    for i in ERA_stable:
        if i==0:
            print("UNSTABLE ERA")




# Analyse global stability of the sketch drawing (qualitative GARY)
def calc_gary_stability(all_boxes):
    global_stability = []
    for q in range(len(all_boxes)):
        supporters_list = deepcopy(graph_supporters[q])
        supportees_list = get_all_supportees(q)
        for i in supportees_list:
            for j in range(len(all_boxes)):
                if (j in graph_supporters[i]) and (j not in get_all_supportees(q)) and j!=q:
                    supporters_list.append(j)
        supportees_list.append(q)
        center_mass_x = 0
        total_mass = 0
        for k in supportees_list:
            center_mass_x = center_mass_x + (all_boxes[k][0]*all_boxes[k][2]*all_boxes[k][3])
            total_mass = total_mass + (all_boxes[k][2]*all_boxes[k][3])
        center_mass_x = center_mass_x / total_mass
        leftmost_support = 99999999
        rightmost_support = -99999999
        for m in supporters_list:
            if m == 9999:
                leftmost_support = -99999999
                rightmost_support = 99999999
            else:
                if all_boxes[m][0]-(all_boxes[m][2]/2.0) < leftmost_support:
                    leftmost_support = all_boxes[m][0]-(all_boxes[m][2]/2.0)
                if all_boxes[m][0]+(all_boxes[m][2]/2.0) > rightmost_support:
                    rightmost_support = all_boxes[m][0]+(all_boxes[m][2]/2.0)
        if (center_mass_x > leftmost_support) and (center_mass_x < rightmost_support):
            global_stability.append(1)
        else:
            global_stability.append(0)
    for s in global_stability:
        if s == 0:
            print("UNSTABLE GLOBAL GARY !!!!!")
            print(global_stability)
            return 0
    return 1




# checks if point (vp,current_y) interescts a box in all boxes, and that this block is below the current one (b)
# returns the box that does intersect the point
def get_point_in_block(vp,current_y,all_boxes,b):
    current_box = all_boxes[b]
    for bb in range(len(all_boxes)):
        box = all_boxes[bb]
        if vp <= box[0]+(box[2]/2.0):
            if vp >= box[0]-(box[2]/2.0):
                if current_y <= box[1]+(box[3]/2.0):
                    if current_y >= box[1]-(box[3]/2.0):
                        if current_box == box:
                            return bb
                        else:
                            if ((box[1]) > (current_box[1]+(current_box[3]/2.0))):                    #below block must have center point below top blocks bottom
                                if ((box[1]-(box[3]/2.0)) > (current_box[1]-(current_box[3]/2.0))):   #below block must have top point below top blocks top point
                                    return bb                 
    return None




# checks if point (vp,current_y) is within a box in all boxes, and that this block is below the current one (b)
def check_point_in_block(vp,current_y,all_boxes,b):
    current_box = all_boxes[b]
    for box in all_boxes:
        if vp <= box[0]+(box[2]/2.0):
            if vp >= box[0]-(box[2]/2.0):
                if current_y <= box[1]+(box[3]/2.0):
                    if current_y >= box[1]-(box[3]/2.0):
                        if current_box == box:
                            return True
                        else:
                            if ((box[1]) > (current_box[1]+(current_box[3]/2.0))):                    #below block must have center point below top blocks bottom
                                if ((box[1]-(box[3]/2.0)) > (current_box[1]-(current_box[3]/2.0))):   #below block must have top point below top blocks top point
                                    return True
    return False




#MORE COMPLEX METHOD
def calc_matthew_stability(all_boxes,valid_supportees):
    global_stability = []
    all_boxes_ori = deepcopy(all_boxes)

    # just make safe area formed by direct supporters of the block:
    safe_areas2 = []
    for b in range(len(all_boxes)):
        if b in valid_supportees:
            leftmost = 99999999
            rightmost = -99999999
            for gg in graph_supporters[b]:
                if gg == 9999:
                    leftmost = -99999999
                    rightmost = 99999999
                else:
                    if (all_boxes[gg][0]-(all_boxes[gg][2]/2.0)) < leftmost:
                        leftmost = (all_boxes[gg][0]-(all_boxes[gg][2]/2.0))
                    if (all_boxes[gg][0]+(all_boxes[gg][2]/2.0)) > rightmost:
                        rightmost = (all_boxes[gg][0]+(all_boxes[gg][2]/2.0))
        safe_areas2.append([leftmost,rightmost])

    all_boxes = deepcopy(all_boxes_ori)
    for b in range(len(all_boxes)):

        if b in valid_supportees:

            new_stable_check = 1
            z = []
            z2 = []
            eligible_supporters = get_all_supporters(b)
            bb = []
            for cc in get_all_supportees(b):
                if cc in valid_supportees:
                    eligible_supporters.append(cc)

            for x in get_all_supportees(b):
                if x in valid_supportees:
                    invalid = 0
                    for y in get_all_supporters(x):
                        if y != b:
                            if y not in eligible_supporters: 
                                invalid = 1
                    if invalid == 0:
                        z.append(x)
            z.append(b)

            center_mass_x = 0
            total_mass = 0
            for k in z:
                center_mass_x = center_mass_x + (all_boxes[k][0]*all_boxes[k][2]*all_boxes[k][3])
                total_mass = total_mass + (all_boxes[k][2]*all_boxes[k][3])
            center_mass_x = center_mass_x / total_mass

            if (center_mass_x < safe_areas2[b][0]) or (center_mass_x > safe_areas2[b][1]):
                if (center_mass_x < safe_areas2[b][0]):
                    pivot_point = safe_areas2[b][0]
                    good_side = "right"
                else:
                    pivot_point = safe_areas2[b][1]
                    good_side = "left"

                for k in get_all_supportees(b):
                    if k in valid_supportees:
                        if k not in z:
                            d = []
                            for n in get_all_supporters(k):
                                block_on_good_side = 0
                                if n in graph_supportees[b]:
                                    if n in valid_supportees:
                                        if good_side == "right":
                                            if (all_boxes[n][0]+(all_boxes[n][2]/2.0)) > pivot_point:
                                                block_on_good_side = 1
                                        if good_side == "left":
                                            if (all_boxes[n][0]-(all_boxes[n][2]/2.0)) < pivot_point:
                                                block_on_good_side = 1

                                if block_on_good_side == 1:
                                    for m in all_boxes:
                                        if m in get_all_supporters(k):
                                            if m in get_all_supportees(n):
                                                if m in valid_supportees:
                                                    if good_side == "right":
                                                        if all_boxes[m][0] > pivot_point:
                                                            d.append(m)
                                                    if good_side == "left":
                                                        if all_boxes[m][0] < pivot_point:
                                                            d.append(m)

                                    if good_side == "right":
                                        if all_boxes[k][0] > pivot_point:
                                            d.append(k)
                                        if all_boxes[n][0] > pivot_point:
                                            d.append(n)
                                    if good_side == "left":
                                        if all_boxes[k][0] < pivot_point:
                                            d.append(k)
                                        if all_boxes[n][0] > pivot_point:
                                            d.append(n)
                            if d != []:
                                max_distance = -99999999
                                best_com = d[0]
                                for ii in range(len(d)):
                                    if abs(all_boxes[ii][0]-pivot_point) > max_distance:
                                        max_distance = abs(all_boxes[ii][0]-pivot_point)
                                        best_com = d[ii]
                                new_block = [all_boxes[best_com][0],0,all_boxes[k][2],all_boxes[k][3]]
                                z2.append(new_block)

                for jj in z:
                    z2.append(all_boxes[jj])

                center_mass_x = 0
                total_mass = 0
                for bob in z2:
                    center_mass_x = center_mass_x + (bob[0]*bob[2]*bob[3])
                    total_mass = total_mass + (bob[2]*bob[3])
                center_mass_x = center_mass_x / total_mass
                if good_side == "right":
                    if center_mass_x < pivot_point:
                        new_stable_check = 0
                if good_side == "left":
                    if center_mass_x > pivot_point:
                        new_stable_check = 0

        global_stability.append(new_stable_check)

    for s in global_stability:
        if s == 0:
            return 0
    return 1




def calc_matthew_stability_ori(all_boxes):
    global_stability = []
    pivot_points = []
    safe_areas2 = []
    for b in range(len(all_boxes)):
        leftmost = 99999999
        rightmost = -99999999
        for gg in graph_supporters[b]:
            if gg == 9999:
                leftmost = -99999999
                rightmost = 99999999
            else:
                if (all_boxes[gg][0]-(all_boxes[gg][2]/2.0)) < leftmost:
                    leftmost = (all_boxes[gg][0]-(all_boxes[gg][2]/2.0))
                if (all_boxes[gg][0]+(all_boxes[gg][2]/2.0)) > rightmost:
                    rightmost = (all_boxes[gg][0]+(all_boxes[gg][2]/2.0))
        safe_areas2.append([leftmost,rightmost])

    all_boxes = deepcopy(all_boxes_ori)
    for b in range(len(all_boxes)):

        new_stable_check = 1
        z = []
        z2 = []
        good_side = None
        eligible_supporters = get_all_supporters(b)+get_all_supportees(b)
        for x in get_all_supportees(b):
            invalid = 0
            for y in get_all_supporters(x):
                if y != b:
                    if y not in eligible_supporters: 
                        invalid = 1
            if invalid == 0:
                z.append(x)
        z.append(b)

        center_mass_x = 0
        total_mass = 0
        for k in z:
            center_mass_x = center_mass_x + (all_boxes[k][0]*all_boxes[k][2]*all_boxes[k][3])
            total_mass = total_mass + (all_boxes[k][2]*all_boxes[k][3])
        center_mass_x = center_mass_x / total_mass


        if (center_mass_x < safe_areas2[b][0]) or (center_mass_x > safe_areas2[b][1]):
            if (center_mass_x < safe_areas2[b][0]):
                pivot_point = safe_areas2[b][0]
                good_side = "right"
            else:
                pivot_point = safe_areas2[b][1]
                good_side = "left"

            for k in get_all_supportees(b):
                if k not in z:


                    if good_side == "right":
                        if all_boxes[k][0]+(all_boxes[k][2]/2.0) > pivot_point:
                            z2.append(k)

                    if good_side == "left":
                        if all_boxes[k][0]-(all_boxes[k][2]/2.0) < pivot_point:
                            z2.append(k)

            for jj in z:
                z2.append(jj)

            supporters_list = deepcopy(graph_supporters[b])
            supportees_list = z2
            for i in supportees_list:
                for j in range(len(all_boxes)):
                    if (j in graph_supporters[i]) and (j not in get_all_supportees(b)) and j!=b:
                        supporters_list.append(j)
            center_mass_x = 0
            total_mass = 0
            for k in supportees_list:
                center_mass_x = center_mass_x + (all_boxes[k][0]*all_boxes[k][2]*all_boxes[k][3])
                total_mass = total_mass + (all_boxes[k][2]*all_boxes[k][3])
            center_mass_x = center_mass_x / total_mass
            leftmost_support = 99999999
            rightmost_support = -99999999
            for m in supporters_list:
                if m == 9999:
                    leftmost_support = -99999999
                    rightmost_support = 99999999
                else:
                    if all_boxes[m][0]-(all_boxes[m][2]/2.0) < leftmost_support:
                        leftmost_support = all_boxes[m][0]-(all_boxes[m][2]/2.0)
                    if all_boxes[m][0]+(all_boxes[m][2]/2.0) > rightmost_support:
                        rightmost_support = all_boxes[m][0]+(all_boxes[m][2]/2.0)
            if (center_mass_x > leftmost_support) and (center_mass_x < rightmost_support):
                new_stable_check = 1
            else:
                new_stable_check = 0

        pivot_points.append(good_side)
        global_stability.append(new_stable_check)

    return [global_stability,pivot_points]




def add_extra_supports(all_boxes,pivot_points,chosen_block):
    new_all_boxes = deepcopy(all_boxes)
    added_block = []
    right_side = 0
    if pivot_points[chosen_block] == "left":
        right_side = 1
    x_position = 0
    if right_side == 1:
        x_position = all_boxes[chosen_block][0]+(all_boxes[chosen_block][2]/2.0) - push_back_distance
    else:
        x_position = all_boxes[chosen_block][0]-(all_boxes[chosen_block][2]/2.0) + push_back_distance
        
    y_position_top = all_boxes[chosen_block][1]+(all_boxes[chosen_block][3]/2.0)
    
    lowest_point = 0
    for i in range(len(all_boxes)):
        if (all_boxes[i][1]+(all_boxes[i][3]/2.0)) > lowest_point:
            lowest_point = (all_boxes[i][1]+(all_boxes[i][3]/2.0))
      
    to_check_hit = []
    for ii in range(len(all_boxes)):
        if all_boxes[ii][1] > all_boxes[chosen_block][1]:
            if ii != 9999:
                to_check_hit.append(all_boxes[ii][1]-(all_boxes[ii][3]/2.0))
            
    to_check_hit = sorted(to_check_hit,reverse=True)
    
    new_to_check_hit = []
    for ppp in range(len(to_check_hit)):
        if to_check_hit[ppp] > (all_boxes[chosen_block][1]+(all_boxes[chosen_block][3]/2.0)):
            new_to_check_hit.append(to_check_hit[ppp])
            
    to_check_hit = new_to_check_hit
        
    y_position_bottom = lowest_point
    found = 0
    while (len(to_check_hit))>0:
        point_to_check = [x_position,to_check_hit[-1]]
        if check_point_in_block(x_position,to_check_hit[-1],all_boxes,chosen_block):
            if found == 0:
                y_position_bottom = to_check_hit[-1]
                found = 1
        to_check_hit.pop()
            
    added_block_x = x_position
    added_block_width = 1
    added_block_height = y_position_bottom-y_position_top
    added_block_y = y_position_top+(added_block_height/2.0)
    added_block = [added_block_x,added_block_y,added_block_width,added_block_height]
    all_boxes.append(added_block)
    
    print("ADDED BLOCK:")
    print(added_block)
    
    return(all_boxes)

    


def find_below_blocks(all_boxes, box):
    below_blocks = []
    for block2 in complete_locations:
        if block2[2]<box[2]:
            if ( (round(box[0],10) <= round((block2[1]+(blocks[str(block2[0])][0]/2)),10))
            and (round(box[1],10) >= round((block2[1]-(blocks[str(block2[0])][0]/2)),10))
            and (round(box[2],10) <= round((block2[2]+(blocks[str(block2[0])][1]/2)),10))
            and (round(box[3],10) >= round((block2[2]-(blocks[str(block2[0])][1]/2)),10)) ):
                below_blocks.append(block2)
    return below_blocks
        
        


#currently doesn't work if multiple structures in image, need to test each structure separately
def calc_other_stability(all_boxes):
    structure_stable = True

    # checks the global stability of level by testing the stability of every block (as peak block)
    highest_block = -1
    highest_com = 99999999
    for block in range(len(all_boxes)):
        if all_boxes[block][1] < highest_com:
            highest_com = all_boxes[block][1]
            highest_block = block

    current_box = [highest_block]
    hit_ground = 0
    if graph_supporters[block] == [9999]:
        hit_ground = 1

    while hit_ground == 0:     # while not at bottom of structure

        support_area = [99999999,0]
        current_com = 0
        total_mass = 0

        supo = []
        for jj in current_box:
            for kk in graph_supporters[jj]:
                if kk not in current_box:
                    supo.append(kk)

        for jj in current_box:
            current_com = current_com + all_boxes[jj][0]*(all_boxes[jj][2]*all_boxes[jj][3])
            total_mass = total_mass + (all_boxes[jj][2]*all_boxes[jj][3])
        current_com = current_com / total_mass
        
        for jj in supo:
            if all_boxes[jj][0] - (all_boxes[jj][2]/2.0) < support_area[0]:
                support_area[0] = all_boxes[jj][0] - (all_boxes[jj][2]/2.0)
            if all_boxes[jj][0] + (all_boxes[jj][2]/2.0) > support_area[1]:
                support_area[1] = all_boxes[jj][0] + (all_boxes[jj][2]/2.0)

        if (current_com >= support_area[1]) or (current_com <= support_area[0]):
            structure_stable = False

        to_add = []
        highest_block = -1
        highest_com = 99999999
        for block in range(len(all_boxes)):
            if block not in current_box:
                if all_boxes[block][1] < highest_com:
                    highest_com = all_boxes[block][1]
                    highest_block = block
        to_add.append(highest_block)

        current_box = current_box + to_add

        if graph_supporters[current_box[-1]] == [9999]:
                hit_ground = 1

    if structure_stable:
        print("STABLE!")
        return 1
    else:
        print("NOT STABLE!")
        return 0




all_boxes_ori_very = deepcopy(all_boxes)
all_stable = 0
while all_stable == 0:
    all_boxes = sorted(all_boxes, key=itemgetter(1), reverse=True)            # sort boxes from bottom to top
    all_boxes_ori = deepcopy(all_boxes)


    #find blocks that touch each other (above and below)
    touching_blocks = []
    touching_lines = []                 
    width_extra = maximum_width_gap_touching
    height_extra = maximum_height_gap_touching
    for i in range(len(all_boxes)):
        current_box = all_boxes[i]
        for j in range(len(all_boxes)):
            box2 = all_boxes[j]
            if ( (current_box[0]-((current_box[2]+width_extra)/2.0) < box2[0]+(box2[2]/2.0)) and
                 (current_box[0]+((current_box[2]+width_extra)/2.0) > box2[0]-(box2[2]/2.0)) and
                 (current_box[1]+((current_box[3]+height_extra)/2.0) > box2[1]-(box2[3]/2.0)) and
                 (current_box[1]-((current_box[3]+height_extra)/2.0) < box2[1]+(box2[3]/2.0)) ):
                if (i != j):
                    if ((current_box[1]) > (box2[1]+(box2[3]/2.0))):                            #below block must have center point below top blocks bottom
                        if ((current_box[1]-(current_box[3]/2.0)) > (box2[1]-(box2[3]/2.0))):   #below block must have top point below top blocks top point
                            touching_blocks.append([i,j])                                       #first box supports the second
                            touching_lines.append([current_box[0]-(current_box[2]/2.0),
                                                   current_box[0]+(current_box[2]/2.0), 
                                                   box2[0]-(box2[2]/2.0),
                                                   box2[0]+(box2[2]/2.0)])                      #bottom block first then top



    new_touching_blocks = []
    new_touching_lines = []
    for i in range(len(all_boxes)):
        for j in range(len(all_boxes)):
            for k in range(len(all_boxes)):
                if [i,j] in touching_blocks:
                    if [i,k] in touching_blocks:
                        if [j,k] in touching_blocks:
                            posie = touching_blocks.index([i,k])
                            touching_blocks.pop(posie)
                            touching_lines.pop(posie)



    # finds the supportees and supporters (direct) for each block
    all_boxes = deepcopy(all_boxes_ori)

    graph_supportees = {}
    graph_supporters = {}

    for i in range(len(all_boxes)):
        graph_supportees[i] = []
        for support in touching_blocks:
            if support[0] == i:
                graph_supportees[i].append(support[1])

    for i in range(len(all_boxes)):
        graph_supporters[i] = []
        for support in touching_blocks:
            if support[1] == i:
                graph_supporters[i].append(support[0])

        if (graph_supporters[i] == []):
            graph_supporters[i] = [9999]    # the ground is represented as block number 9999

    all_boxes = deepcopy(all_boxes_ori)
    check_local_stability(all_boxes)

    all_boxes = deepcopy(all_boxes_ori)
    ERA_relations = find_era_relations()

    all_boxes = deepcopy(all_boxes_ori)
    calc_era_stability(all_boxes, ERA_relations, touching_blocks)

    all_boxes = deepcopy(all_boxes_ori)
    testg = calc_gary_stability(all_boxes)

    if testg == 0:
        GARY_INITIAL = 0

    all_boxes = deepcopy(all_boxes_ori)
    testg = calc_other_stability(all_boxes)

    if testg == 0:
        OTHER_INITIAL = 0
        
    # Analyse global stability of the sketch drawing (new qualitative method)
    all_boxes = deepcopy(all_boxes_ori)

    chosen_block = 99999999
    global_stable_sketch = 1
    both = calc_matthew_stability_ori(all_boxes)
    global_stability = both[0]
    pivot_points = both[1]
    for s in global_stability:
        if s == 0:
            global_stable_sketch = 0
            print("GLOBALLY UNSTABLE MATTHEW")
            MATTHEW_INITIAL = 0
            print(both)
            
    if (global_stable_sketch == 0):
        for j in range(len(global_stability)):
            if global_stability[j] == 0:
                chosen_block = j

        all_boxes = add_extra_supports(all_boxes,pivot_points,chosen_block)

    else:
        all_stable = 1
        
    if add_extra_blocks_to_make_stable == 0:
        all_stable = 1
    else:
        all_boxes_ori = deepcopy(all_boxes)




def merge_groups(groupings):
    to_merge = []
    for g1 in range(len(groupings)):
        for g2 in range(len(groupings)):
            if (g1 != g2):
                for g1_block in groupings[g1]:
                    for g2_block in groupings[g2]:
                        if g1_block == g2_block:
                            to_merge.append([g1,g2])
                            return to_merge
    return to_merge




def remove_groupings(groupings):
    to_remove = []
    for g1 in range(len(groupings)):
        for g2 in range(len(groupings[g1])):
            for g3 in range(len(groupings[g1])):
                if (g2<g3):
                    if groupings[g1][g2] == groupings[g1][g3]:
                        to_remove.append([g1,g3])
                        return to_remove
    return to_remove




# splits block sets into groupings that must have the same height
all_boxes = deepcopy(all_boxes_ori)
groupings = []
no_change1 = 0
if check_groups==1:
    no_change1 = 1
    old_groupings = deepcopy(groupings)
    for i in range(len(all_boxes)):
        for j in range(len(all_boxes)):
            if (i < j):

                # checks if i and j share a direct supporter
                direct_supporter = 0
                for b1 in graph_supporters[i]:
                    for b2 in graph_supporters[j]:
                        if (b1==b2):
                            direct_supporter = 1
       
                if (direct_supporter == 1):     # check if i and j share a supportee
                    for k in range(len(all_boxes)):
                        if len(find_all_paths(graph_supportees,i,k)) > 0:
                            if len(find_all_paths(graph_supportees,j,k)) > 0:
                                groupings.append([])
                                for aa in find_all_paths(graph_supportees,i,k):
                                    aa.pop()
                                    groupings[-1].append(aa)
                                for bb in find_all_paths(graph_supportees,j,k):
                                    bb.pop()
                                    groupings[-1].append(bb)


    # merge groups togethor (originally the same indentation level as the above paragraph of code)
    cleverMergeLists(groupings)
                                
    #remove duplicates    
    no_change3 = 0
    while (no_change3 == 0):
        to_remove = remove_groupings(groupings)
        if len(to_remove) == 0:
            no_change3 = 1
        else:
            del groupings[to_remove[0][0]][to_remove[0][1]]

    if sorted(old_groupings) != sorted(groupings):
        no_change1=0




# make all single blocks in groups the average height of all single blocks in same group
all_boxes = deepcopy(all_boxes_ori)
if (average_single_block_groups_heights==1):
    for g in groupings:
        to_average = []
        average_height = 0
        total_height = 0
        for block_set in g:
            if len(block_set)==1:
                to_average.append(block_set[0])
        if len(to_average)>0:
            for b in to_average:
                total_height = total_height+all_boxes[b][3]
            average_height = total_height/float(len(to_average))
            for b in to_average:
                all_boxes[b][3] = average_height




if (use_similarity_grouping == 1):
    close_distance = largest_value*2
    blocks_same = []
    for i in range(len(all_boxes)):
        for j in range(len(all_boxes)):
            same_shape = 0
            close = 0
            no_inbetween = 1
            if i != j:
                if all_boxes[i][0] < (all_boxes[j][0] + all_boxes[j][0]*error_percentage_shape):
                    if all_boxes[i][0] > (all_boxes[j][0] - all_boxes[j][0]*error_percentage_shape):
                        if all_boxes[i][2] < (all_boxes[j][2] + all_boxes[j][2]*error_percentage_shape):
                            if all_boxes[i][2] > (all_boxes[j][2] - all_boxes[j][2]*error_percentage_shape):
                                if all_boxes[i][3] < (all_boxes[j][3] + all_boxes[j][3]*error_percentage_shape):
                                    if all_boxes[i][3] > (all_boxes[j][3] - all_boxes[j][3]*error_percentage_shape):
                                        same_shape = 1

                elif all_boxes[i][1] < (all_boxes[j][1] + all_boxes[j][1]*error_percentage_shape):
                    if all_boxes[i][1] > (all_boxes[j][1] - all_boxes[j][1]*error_percentage_shape):
                        if all_boxes[i][2] < (all_boxes[j][2] + all_boxes[j][2]*error_percentage_shape):
                            if all_boxes[i][2] > (all_boxes[j][2] - all_boxes[j][2]*error_percentage_shape):
                                if all_boxes[i][3] < (all_boxes[j][3] + all_boxes[j][3]*error_percentage_shape):
                                    if all_boxes[i][3] > (all_boxes[j][3] - all_boxes[j][3]*error_percentage_shape):
                                        same_shape = 1

                if all_boxes[i][0] < (all_boxes[j][0] + close_distance):
                    close = 1
                if all_boxes[i][0] > (all_boxes[j][0] - close_distance):
                    close = 1
                if all_boxes[i][1] < (all_boxes[j][1] + close_distance):
                    close = 1
                if all_boxes[i][1] > (all_boxes[j][1] - close_distance):
                    close = 1

                for k in range(len(all_boxes)):
                    k_top = all_boxes[k][1] + (all_boxes[k][3]/2.0)
                    k_bottom = all_boxes[k][1] - (all_boxes[k][3]/2.0)
                    k_left = all_boxes[k][0] - (all_boxes[k][2]/2.0)
                    k_right = all_boxes[k][0] + (all_boxes[k][2]/2.0)
                    i_top = all_boxes[i][1] + (all_boxes[i][3]/2.0)
                    i_bottom = all_boxes[i][1] - (all_boxes[i][3]/2.0)
                    i_left = all_boxes[i][0] - (all_boxes[i][2]/2.0)
                    i_right = all_boxes[i][0] + (all_boxes[i][2]/2.0)
                    j_top = all_boxes[j][1] + (all_boxes[j][3]/2.0)
                    j_bottom = all_boxes[j][1] - (all_boxes[j][3]/2.0) 
                    j_left = all_boxes[j][0] - (all_boxes[j][2]/2.0)
                    j_right = all_boxes[j][0] + (all_boxes[j][2]/2.0) 
                    
                    if (k_top > i_bottom) and (k_top > j_bottom) and (k_bottom < i_top) and (k_bottom < j_top) and (all_boxes[k][0] > all_boxes[i][0]) and (all_boxes[k][0] < all_boxes[j][0]):
                        no_inbetween = 0
                    if (k_right > i_left) and (k_right > j_left) and (k_left < i_right) and (k_left < j_right) and (all_boxes[k][1] > all_boxes[i][1]) and (all_boxes[k][1] < all_boxes[j][1]):
                        no_inbetween = 0

                if (no_inbetween==1 and close==1 and same_shape==1):
                    blocks_same.append([i,j])




if ((average_same_block_groups_heights==1) and (use_similarity_grouping == 1)):
    blocks_same2 = deepcopy(blocks_same)
    no_change2 = 0
    while(no_change2 == 0):
        to_merge = []
        for g1 in range(len(blocks_same2)):
            for g1_block in blocks_same2[g1]:
                for g2 in range(len(blocks_same2)):
                    for g2_block in blocks_same2[g2]:
                        if (g1 != g2):
                            if g1_block == g2_block:
                                to_merge.append([g1,g2])
        if len(to_merge) == 0:
            no_change2 = 1
        else:
            blocks_same2[to_merge[0][0]] = blocks_same2[to_merge[0][0]]+blocks_same2[to_merge[0][1]]
            blocks_same2.pop(to_merge[0][1])
                    
            
    #remove duplicates       
    no_change3 = 0
    while (no_change3 == 0):
        to_remove = []
        no_change3=1
        for g1 in range(len(blocks_same2)):
            for g2 in range(len(blocks_same2[g1])):
                for g3 in range(len(blocks_same2[g1])):
                    if (g2<g3):
                        if blocks_same2[g1][g2] == blocks_same2[g1][g3]:
                            no_change3=0
                            to_remove.append([g1,g3])

        if (no_change3 == 0):
            del blocks_same2[to_remove[0][0]][to_remove[0][1]]


    # make same average height
    for g in blocks_same2:
        to_average = []
        average_height = 0
        total_height = 0
        for blockz in g:
            to_average.append(blockz)
        if len(to_average)>0:
            for b in to_average:
                total_height = total_height+all_boxes[b][3]
            average_height = total_height/float(len(to_average))
            for b in to_average:
                all_boxes[b][3] = average_height




# adds composite blocks to set of possible block types, made up of multiple smaller blocks
# can also rearrange the ordering of this sub-blocks to create even more possible options
if composite_blocks_allowed == 1:
    specials = {}
    horizontal = [5,6,8,10,12]
    counter = 14
    for i in range(max_composite_block_width):
        for j in horizontal:
            new_block_width = (2.06*i) + blocks[str(j)][0]
            height_counter = 0.22
            height_num = 1
            while height_counter < new_block_width*2.0:
                pos_j = deepcopy(i)

                if rearrange_special_block_order == 1:

                    while pos_j >= 0:
                        blocks[str(counter)] = [round(new_block_width,2),round(height_counter,2)]
                        block_names[str(counter)] = "special"
                        specials[str(counter)] = [i,j,height_num,pos_j]
                        counter = counter + 1
                        pos_j = pos_j - 1
                    height_counter = round(height_counter + 0.22,2)
                    height_num = height_num+1
                
                else:
                    blocks[str(counter)] = [round(new_block_width,2),round(height_counter,2)]
                    block_names[str(counter)] = "special"
                    specials[str(counter)] = [i,j,height_num,pos_j]
                    counter = counter + 1
                    height_counter = round(height_counter + 0.22,2)
                    height_num = height_num+1




# divide the size and position of all blocks by the scale factor
scale_factor = 1
if scaling_method == 0:
    scale_factor = largest_value/2.06                   # BIG APPROACH

if scaling_method == 1:
    scale_factor = smallest_value/0.22                  # SMALL APPROACH

if scaling_method == 2:
    middle_block_size = (largest_value+smallest_value)/2.0
    scale_factor = middle_block_size/1.14               # MIDDLE APPROACH

if scaling_method == 3:
    scale_factor = mean_size/actual_block_mean          # MEAN APPROACH (0.667)

if scaling_method == 4:
    scale_factor = median_size/actual_block_median      # MEDIAN APPROACH (0.43)
         
all_boxes2 = []
for box in all_boxes:
    box2 = []
    box2.append(box[0]/scale_factor)
    box2.append(box[1]/scale_factor)
    box2.append(box[2]/scale_factor)
    box2.append(box[3]/scale_factor)
    all_boxes2.append(box2)

block_order= []
for i in range(len(all_boxes2)):
    block_order.append(i)




# re-order list so that blocks are place straight after their direct supporters (or as close as possible to straight after, lower blocks get priority)

# re-orders blocks from being supporters before supportees (closer togethor) rather than top to bottom
# add very bottom block to list
# add supporter of block to list (only if all supporters of this block are present)
# if not all supporters are present, then add all supportees of this block to the list

if order_blocks_smart == 1:
    block_order = [0]
    block_order2 = [0]
    while(len(block_order) < len(all_boxes2)):
        added_block = 0
        for i in reversed(block_order):
            for j in graph_supportees[i]:
                if j not in block_order:
                    if j not in block_order2:
                        all_supporters = 1
                        for k in graph_supporters[j]:
                            if k not in block_order:
                                all_supporters = 0
                                check_order = []
                                to_check = [k]
                                while(len(to_check)>0):
                                    value_checking = to_check.pop()
                                    check_order.append(value_checking)
                                    for yup in graph_supporters[value_checking]:
                                        if yup != 9999:
                                            to_check.append(yup)
                                for kk in check_order:
                                    if (kk not in block_order2) and (added_block == 0):
                                        add_me = 1
                                        for gg in graph_supporters[kk]:
                                            if (gg not in block_order2) and gg!=9999:
                                                add_me = 0
                                        if add_me == 1:
                                            block_order2.append(kk)
                                            added_block = 1

                        if (all_supporters == 1) and (added_block == 0):
                            block_order2.append(j)
                            added_block = 1

        if (block_order == block_order2):
            for rem in range(len(all_boxes2)):
                if rem not in block_order2:
                    block_order2.append(rem)

        block_order = deepcopy(block_order2)




# find block type with most similar size to each block
block_keys = numpy.empty((len(all_boxes2), 0)).tolist()
already_tried = [[]]
count_loops = 0
ori_blocks3 = deepcopy(all_boxes2)
all_done = 0
while (all_done == 0) and (count_loops < 10000):

    current_index = 0
    for qqq in block_keys:
        if qqq != []:
            current_index = current_index + 1

    current_box = block_order[current_index]
    box = ori_blocks3[current_box]

    count_loops = count_loops+1
    if count_loops % 1000 == 0:
        print("generating...")          # prints every 1000 loops
    

    # choose a block type for the next block to be added
    # based on the sum of the squared differences between their widths and heights (width_dif^2 + height_dif^2)
    width = box[2]
    height = box[3]
    best_difference = 99999999
    best_name = ""
    for key,value in blocks.items():
        width_difference = abs(width-value[0])
        height_difference = abs(height-value[1])
        total_difference = width_difference**2 + height_difference**2
        if int(key) > original_number_blocks:
            total_difference = total_difference*composite_block_penalty_picking
        if (best_difference > total_difference):
            if (key not in already_tried[-1]):
                best_difference = total_difference
                best_name = key
    block_keys[current_box] = best_name
    already_tried[-1].append(best_name)


    # move block to correct height (based on supporting block height)                                        
    if graph_supporters[current_box] == [9999]:
        new = []
        new.append(ori_blocks3[current_box][0])
        new.append(ground+(blocks[block_keys[current_box]][1]/2.0))
        new.append(blocks[block_keys[current_box]][0])
        new.append(blocks[block_keys[current_box]][1])
    else:
        new = []
        new.append(ori_blocks3[current_box][0])
        new.append(all_boxes2[graph_supporters[current_box][0]][1]+
                    (blocks[block_keys[graph_supporters[current_box][0]]][1]/2.0)+      # error might happen here if structure not possible
                    (blocks[block_keys[current_box]][1]/2.0))
        new.append(blocks[block_keys[current_box]][0])
        new.append(blocks[block_keys[current_box]][1])
    all_boxes2[current_box] = new

        
    # CHECK THAT BLOCK JUST ADDED TO BLOCK KEYS DOESNT VIOLATE ANY RULES
    # if it does then pop the key off block_keys
    # do iteratively, removing previous block if no block types for the current block are possible

    must_pop = 0
    if use_similarity_grouping:
        for tim in blocks_same:
            if tim[0] == current_box:
                if block_keys[tim[1]] != []:
                    if block_keys[tim[0]] != block_keys[tim[1]] :
                        must_pop = 1
            if tim[1] == current_box:
                if block_keys[tim[0]] != []:
                    if block_keys[tim[0]] != block_keys[tim[1]] :
                        must_pop = 1


    # ensures that chosen block type is the right height to fulfil all grouping requirments
    # outside of the horizontal movement shift option as that won't help correct this
    if (check_groups == 1) and must_pop==0:
        for g in groupings:
            height_set = 0
            for block_set1 in g:
                valid = 1
                for n in block_set1:
                    if (block_keys[n]==[]):
                        valid = 0
                if valid == 1:
                    height_set2 = 0
                    for nn in block_set1:
                        height_set2 += blocks[block_keys[nn]][1]
                    if height_set == 0:
                        height_set = height_set2
                    else:
                        if abs(height_set - height_set2) > height_error_allowed_groups:
                            must_pop = 1


    # Check if comoposite block is locally stable (all blocks that make it up are supported)
    if (check_composite_block_stability == 1) and (composite_blocks_allowed == 1) and must_pop==0:
        block_num_special = block_keys[current_box]
        i = block_keys[current_box]
        j = all_boxes2[current_box]
        if int(block_num_special) > original_number_blocks:
            info = specials[i]
            total_width = round((2.06*info[0])+blocks[str(info[1])][0],2)
            total_height = info[2]*0.22
            positions_long = []             
            position_extra = []
            added_j = 0
            current_pos = j[0]-(total_width/2.0) 
            y_pos = j[1] - (total_height/2.0) + 0.11
            for a in range(info[0]):
                if a == info[3]:
                    added_j = 1
                    current_pos = current_pos + (blocks[str(info[1])][0]/2.0)
                    position_extra = current_pos
                    current_pos = current_pos + (blocks[str(info[1])][0]/2.0)
                current_pos = current_pos + 1.03
                positions_long.append(current_pos)
                current_pos = current_pos + 1.03
            if added_j == 0:
                current_pos = current_pos + (blocks[str(info[1])][0]/2.0)
                position_extra = current_pos

            all_boxes_special = []
            block_keys_special = []
            for iii in range(len(positions_long)):
                all_boxes_special.append([positions_long[iii],y_pos,2.06,0.22])
                block_keys_special.append(12)
            all_boxes_special.append([position_extra,y_pos,blocks[str(info[1])][0],0.22])
            block_keys_special.append(info[1])
    
            # check local stability
            width_error_allowed_local_composite = 0.0
            for ii in range(len(all_boxes_special)):
                left_support = 0
                right_support = 0
                box = all_boxes_special[ii]
                for jj in graph_supporters[current_box]:
                    if jj == 9999:
                        left_support = 1
                        right_support = 1
                    else:
                        box2 = all_boxes2[jj]
                        box2_left = box2[0]-((blocks[block_keys[jj]][0])/2.0)
                        box2_right = box2[0]+((blocks[block_keys[jj]][0])/2.0)
                        if box2_left < box[0] + width_error_allowed_local_composite:
                            if box2_right > (box[0] - (box[2]/2.0)):
                                left_support = 1
                        if box2_right > box[0] - width_error_allowed_local_composite:
                            if box2_left < (box[0] + (box[2]/2.0)):
                                right_support = 1
                if left_support == 0:
                    must_pop = 1
                if right_support == 0:
                    must_pop = 1 


    if must_pop == 0:
        tried_all_moving = 0

        while (tried_all_moving==0):
            must_pop = 0

            # ensures the chosen block does not overlap any other already chosen blocks
            if (check_overlap == 1) and must_pop==0:
                width_error_allowed_overlap = 0.0
                for i in range(len(all_boxes2)):
                    if (block_keys[i]!=[]) and (i!=current_box):                                    
                        box_width = blocks[best_name][0]-width_error_allowed_overlap
                        box_height = blocks[best_name][1]-height_error_allowed_overlap
                        box2 = all_boxes2[i]
                        box2_width = blocks[block_keys[i]][0]-width_error_allowed_overlap           
                        box2_height = blocks[block_keys[i]][1]-height_error_allowed_overlap
                        if ( (all_boxes2[current_box][0]-(box_width/2.0) < box2[0]+(box2_width/2.0)) and
                             (all_boxes2[current_box][0]+(box_width/2.0) > box2[0]-(box2_width/2.0)) and
                             (all_boxes2[current_box][1]+(box_height/2.0) > box2[1]-(box2_height/2.0)) and
                             (all_boxes2[current_box][1]-(box_height/2.0) < box2[1]+(box2_height/2.0)) ):
                            must_pop = 1


            # ensures that chosen block type is wide enough to be supported by all direct supporter blocks
            if (check_all_supporters == 1) and must_pop==0:
                for i in graph_supporters[current_box]:
                    if (i < 9999):
                        test_box = all_boxes2[i]
                        if (all_boxes2[current_box][0]-(blocks[best_name][0]/2.0) + required_support_amount) > (test_box[0]+(blocks[block_keys[i]][0]/2.0)):
                            must_pop = 1
                        if (all_boxes2[current_box][0]+(blocks[best_name][0]/2.0) - required_support_amount) < (test_box[0]-(blocks[block_keys[i]][0]/2.0)):
                            must_pop = 1

                    
            # CHECK ERA RELATIONS (OPTIONAL)    NOT SURE IF WORKS 100% BUT SHOULDN'T BE USED ANYWAY AS PREVENTS STABILITY CORRECTION AND VERY RESTRICTIVE
            if (check_era_relations == 1) and must_pop==0:
                width_extra_era = 0.06
                height_extra_era = 0.02
                touching_blocks2 = []
                touching_lines2 = []   
                era_relations2 = []              
                for i in range(len(all_boxes2)):
                    if block_keys[i] != []:
                                       
                        current_box2 = all_boxes2[i]
                        current_box2[2] = current_box2[2]+width_extra_era         
                        current_box2[3] = current_box2[3]+height_extra_era
                        for j in range(len(all_boxes2)):
                            if block_keys[j] != []:
                            
                                box2 = all_boxes2[j]
                                if ( (current_box2[0]-(current_box2[2]/2.0) < box2[0]+(box2[2]/2.0)) and
                                     (current_box2[0]+(current_box2[2]/2.0) > box2[0]-(box2[2]/2.0)) and
                                     (current_box2[1]+(current_box2[3]/2.0) > box2[1]-(box2[3]/2.0)) and
                                     (current_box2[1]-(current_box2[3]/2.0) < box2[1]+(box2[3]/2.0)) ):
                                    if (i != j):
                                        if ((current_box2[1]) > (box2[1]+(box2[3]/2.0))):                            
                                            if ((current_box2[1]-(current_box2[3]/2.0)) > (box2[1]-(box2[3]/2.0))):   
                                                touching_blocks2.append([j,i])                                       
                                                touching_lines2.append([box2[0]-(box2[2]/2.0),
                                                                        box2[0]+(box2[2]/2.0),
                                                                        current_box2[0]-(current_box2[2]/2.0),
                                                                        current_box2[0]+(current_box2[2]/2.0)])                    

                for pairin in range(len(touching_blocks)):
                    if block_keys[touching_blocks[pairin][0]] != []:
                        if block_keys[touching_blocks[pairin][1]] != []:
                            if touching_blocks[pairin] not in touching_blocks2:
                                must_pop=1

                for line2 in touching_lines2:
                    era_relations2.append(find_era_relation(line2))
                
                for ori1 in range(len(touching_blocks)):
                    if block_keys[touching_blocks[ori1][0]]!=[]: 
                        first_block = touching_blocks[ori1][0]
                        if block_keys[touching_blocks[ori1][1]]!=[]: 
                            second_block = touching_blocks[ori1][1]
                            correct_index_new = 99999999
                            for new1 in range(len(touching_blocks2)):
                                if touching_blocks2[new1] == [first_block,second_block]:
                                    correct_index_new = new1
                            if correct_index_new < 99999999:
                                if ERA_relations[ori1] != era_relations2[correct_index_new][0]:
                                    must_pop = 1
    

            # check if structure has local stability            
            # BETTER TO CHECK GLOBAL STABILITY UNLESS TRYING TO BE FAST
            if (check_local_stability == 1) and must_pop==0:
                width_error_allowed_local = 0.0
                for i in range(len(all_boxes2)):
                    if (block_keys[i]!=[]):
                        left_support = 0
                        right_support = 0
                        box = all_boxes2[i]
                        for j in graph_supporters[i]:
                            if j == 9999:
                                left_support = 1
                                right_support = 1
                            else:
                                box2 = all_boxes2[j]
                                box2_left = box2[0]-((blocks[block_keys[j]][0])/2.0)
                                box2_right = box2[0]+((blocks[block_keys[j]][0])/2.0)
                                if box2_left < box[0] + width_error_allowed_local:
                                    left_support = 1
                                if box2_right > box[0] - width_error_allowed_local:
                                    right_support = 1
                        if left_support == 0:
                            must_pop = 1
                        if right_support == 0:
                            must_pop = 1 


            # check if structure has global stability   
            if (check_global_stability == 1) and must_pop==0:
                stable_global = 0
                valid_supportees = []
                new_joint_all_boxes = []
                if check_global_stability_method == 1:
                    new_joint_all_boxes = deepcopy(all_boxes2)
                    for k in range(len(block_keys)):
                        if block_keys[k] != []:
                            valid_supportees.append(k)
                elif check_global_stability_method == 2:
                    for k in range(len(all_boxes2)):
                        valid_supportees.append(k)
                        if block_keys[k] != []:
                            new_joint_all_boxes.append(all_boxes2[k])
                        else:
                            new_joint_all_boxes.append(ori_blocks3[k])
                else:
                    print ("ERROR!! WRONG CHECK GLOBAL STABILITY METHOD")
                stable_global = calc_matthew_stability(new_joint_all_boxes,valid_supportees)
                if stable_global == 0:
                    must_pop = 1


            # move sideways if selected as viable response option
            if must_pop == 1:
                if len(moves_to_try)==0:
                    tried_all_moving=1
                elif shift_blocks_sideways==0:
                    tried_all_moving=1
                else:
                    all_boxes2[current_box][0] = all_boxes2[current_box][0]+moves_to_try[-1]
                    moves_to_try.pop()
            else:
                tried_all_moving = 1


    # block fails one or more requirments so remove it and try again
    if must_pop == 1:    
        block_keys[current_box]=[]
        # if already tried all block types then remove the block AND the previous block
        if (limit_number_block_type_changes == 1) and (len(blocks) > max_number_block_type_changes):
            while len(already_tried[-1]) == max_number_block_type_changes:          
                current_index = current_index-1
                block_keys[block_order[current_index]]=[]
                already_tried.pop()
        else:
            while len(already_tried[-1]) == len(blocks):          
                current_index = current_index-1
                block_keys[block_order[current_index]]=[]
                already_tried.pop()
    else:
        already_tried.append([])

    all_done=1
    for qqq in block_keys:
        if qqq == []:
            all_done=0




if (count_loops >= 10000):
    print("generating structure took too long, suggest trying a different scale_calculation_option")



#calculate measure of difference betweent the original sketch and the generated structure (average percentage ratio difference)
avg_ratio_error_score = 0
for i in range(len(all_boxes_ori)):
    ratio_ori = all_boxes_ori[i][3]/all_boxes_ori[i][2]
    ratio_new = blocks[block_keys[i]][1]/blocks[block_keys[i]][0]
    avg_ratio_error_score = avg_ratio_error_score + (abs(ratio_ori-ratio_new)/((ratio_ori+ratio_new)/2.0))
avg_ratio_error_score = avg_ratio_error_score/len(all_boxes_ori)
print("AVG RATIO ERROR:")
print(avg_ratio_error_score)


avg_mean_error_score = 0
old_mean_area = deepcopy(mean_area)
new_mean_area = 0
for i in range(len(all_boxes_ori)):
    new_mean_area = new_mean_area + (blocks[block_keys[i]][1]*blocks[block_keys[i]][0])
new_mean_area = new_mean_area / len(all_boxes_ori)
old_scale = old_mean_area/100.0
new_scale = new_mean_area/100.0
for i in range(len(all_boxes_ori)):
    area_old = all_boxes_ori[i][3]*all_boxes_ori[i][2]
    area_new = ((blocks[block_keys[i]][1]*blocks[block_keys[i]][0]))
    
    area_old = area_old / old_scale
    area_new = area_new / new_scale

    avg_mean_error_score = avg_mean_error_score + abs((area_old)-(area_new))
avg_mean_error_score = avg_mean_error_score
avg_mean_error_score = avg_mean_error_score/len(all_boxes_ori)
print("AVG MEAN AREA ERROR:")
print(avg_mean_error_score)


avg_location_error_score = 0
old_mean_area = deepcopy(mean_area)
new_mean_area = 0
for i in range(len(all_boxes_ori)):
    new_mean_area = new_mean_area + (blocks[block_keys[i]][1]*blocks[block_keys[i]][0])
new_mean_area = new_mean_area / len(all_boxes_ori)
old_scale = sqrt(old_mean_area)/100.0
new_scale = sqrt(new_mean_area)/100.0
center_mass_new_x = 0
center_mass_new_y = 0
total_mass_new = 0
for i in range(len(all_boxes_ori)):
    box = all_boxes2[i]
    center_mass_new_x = center_mass_new_x + (box[0]*box[2]*box[3])
    center_mass_new_y = center_mass_new_y + (box[1]*box[2]*box[3])
    total_mass_new = total_mass_new + (box[2]*box[3])
center_mass_new_x = center_mass_new_x / total_mass_new
center_mass_new_y = center_mass_new_y / total_mass_new
for i in range(len(all_boxes_ori)):
    position_old_x = abs(all_boxes_ori[i][0]-center_mass_ori_x)
    position_old_y = abs(all_boxes_ori[i][1]-center_mass_ori_y)
    position_new_x = abs(all_boxes2[i][0]-center_mass_new_x)
    position_new_y = abs(all_boxes2[i][1]-center_mass_new_y)

    position_old_x = position_old_x / (old_scale)
    position_old_y = position_old_y / (old_scale)
    position_new_x = position_new_x / (new_scale)
    position_new_y = position_new_y / (new_scale)

    distance = sqrt( (abs(position_old_x-position_new_x)*abs(position_old_x-position_new_x)) + (abs(position_old_y-position_new_y)*abs(position_old_y-position_new_y)) )
    avg_location_error_score = avg_location_error_score + distance
avg_location_error_score = avg_location_error_score
avg_location_error_score = avg_location_error_score/len(all_boxes_ori)
print("AVG LOCATION AREA ERROR:")
print(avg_location_error_score)


penalty_composite = 0.0
number_composite = 0.0
total_number_blocks = len(block_keys)
for i in range(len(block_keys)):
    if int(block_keys[i]) > original_number_blocks:
        number_composite = number_composite + 1
ratio_composite = number_composite/total_number_blocks
penalty_composite = composite_block_penalty_end*ratio_composite
print("PENALTY COMPOSITE:")
print(penalty_composite)


penalty_extra = 0
penalty_weight = 1.0
for i in range(len(all_boxes2)):
    if i >= len(all_boxes_ori_very):
        penalty_extra = penalty_extra + ((blocks[block_keys[i]][1]*blocks[block_keys[i]][0]) / (new_scale))
print("PENALTY EXTRA:")
print(penalty_extra)


print("FINAL ERROR SCORE:")
print((avg_ratio_error_score*avg_mean_error_score*avg_location_error_score)+penalty_composite+penalty_extra)       # Not normlaised




# flip y_axis direction (upwards is now positive rather than negative)
all_boxes3 = []
need_move_up = 0
for i in all_boxes2:
    new = []
    new.append(i[0])
    new.append(i[1]*-1)
    new.append(i[2])
    new.append(i[3])
    all_boxes3.append(new)



        
# move blocks to correct height (needs to be done again after flipping y-axis)
for i in range(len(all_boxes3)):
    if graph_supporters[i] == [9999]:
        new = []
        new.append(all_boxes3[i][0])
        new.append(ground+(blocks[block_keys[i]][1]/2.0))
        new.append(all_boxes3[i][2])
        new.append(all_boxes3[i][3])
    else:
        new = []
        new.append(all_boxes3[i][0])
        new.append(all_boxes3[graph_supporters[i][0]][1]+
                    (blocks[block_keys[graph_supporters[i][0]]][1]/2.0)+
                    (blocks[block_keys[i]][1]/2.0))
        new.append(all_boxes3[i][2])
        new.append(all_boxes3[i][3])
    all_boxes3[i] = new
    
all_boxes4 = all_boxes3




# write XML
number_birds=3
f = open("level-4.xml", "w")
f.write('<?xml version="1.0" encoding="utf-16"?>\n')
f.write('<Level width ="2">\n')
f.write('<Camera x="0" y="2" minWidth="20" maxWidth="30">\n')
f.write('<Birds>\n')
for i in range(number_birds):
    f.write('<Bird type="BirdRed"/>\n')
f.write('</Birds>\n')
f.write('<Slingshot x="-8" y="-2.5">\n')
f.write('<GameObjects>\n')

for index in range(len(all_boxes4)):
    i = block_keys[index]
    j = all_boxes4[index]
    
    if int(i) > original_number_blocks:
        rotation = 0
        info = specials[i]
        total_width = round((2.06*info[0])+blocks[str(info[1])][0],2)
        total_height = info[2]*0.22
        y_pos = j[1] - (total_height/2.0) + 0.11

        pos_j = info[3]
        for jj in range(info[2]):
            positions_long = []             
            position_extra = []
            added_j = 0
            current_pos = j[0]-(total_width/2.0) 
            
            for a in range(info[0]):
                if a == pos_j:
                    added_j = 1
                    current_pos = current_pos + (blocks[str(info[1])][0]/2.0)
                    position_extra = current_pos
                    current_pos = current_pos + (blocks[str(info[1])][0]/2.0)
                current_pos = current_pos + 1.03
                positions_long.append(current_pos)
                current_pos = current_pos + 1.03
            if added_j == 0:
                current_pos = current_pos + (blocks[str(info[1])][0]/2.0)
                position_extra = current_pos
            for aa in range(len(positions_long)):
                f.write('<Block type="RectBig" material="stone" x="%s" y="%s" rotation="0" />\n' % (str(positions_long[aa]), str(y_pos)))
            f.write('<Block type="%s" material="stone" x="%s" y="%s" rotation="0" />\n' % (str(block_names[str(info[1])]),str(position_extra), str(y_pos)))
            y_pos = y_pos + 0.22 

            if composite_block_interweaving == 1:
                pos_j = pos_j + 1
                if pos_j > info[0]:
                    pos_j = 0

    else:
        rotation = 0
        if (int(i) in (3,7,9,11,13)):
            rotation = 90
        f.write('<Block type="%s" material="stone" x="%s" y="%s" rotation="%s" />\n' % (block_names[str(i)],str(j[0]), str(j[1]), str(rotation)))

f.write('</GameObjects>\n')
f.write('</Level>\n')

f.close()



