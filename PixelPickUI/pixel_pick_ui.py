import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker, VPacker, OffsetImage
import matplotlib.patches as mpatches
import numpy as np
import cv2
import io
import matplotlib.pyplot as plt
from skimage import transform
import csv
import pandas as pd

# Returns true if a pixel has already been labelled 
def is_pixel_in_database(video_name, frame, x, y):
    # Opens database file
    with open('pixel_database.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        # Check every row in the database to see if it matches the pixel specified by this functions parameters
        for row in spamreader:
            if row[0] == video_name and row[1] == str(frame) and row[2] == str(x) and row[3] == str(y):
                return True
    return False

# Draws a circle over pixels we want to annotate
def color_points(x, loc_y, loc_x, fc=(0, 128, 192), ec=(255, 255, 255), ms=40, es=10):
    # Draws white outline circle
    x = cv2.circle(x, (loc_x, loc_y), ms + es, color=ec, thickness=-1)
    # Draws inner blue/red circle
    x = cv2.circle(x, (loc_x, loc_y), ms, color=fc, thickness=-1)
    return x

def annotate(arr):
    # Display window
    cv2.namedWindow('image')
    cv2.imshow('image', arr)

    # Valid keys that can be pressed
    alphabet_l = ['g','n']

    # Get a key press from the user and respond accordingly
    while True:
        k = cv2.waitKey()
        if chr(k).lower() not in alphabet_l:
            print(f"\nYou've clicked a wrong character: {chr(k)}\n")

        elif chr(k).lower() in alphabet_l:
            break
        
        elif k == 27: # esc key
            break

    # Terminate window and return label
    cv2.destroyAllWindows()
    return chr(k)

def plot_all_pixels(img, pixel_coords, idx):
    h, w, _ = img.shape
    dim = min(h,w)
    ms=max(int(dim/75), 4)
    es=max(int(dim/300), 1)
    for i in range(len(pixel_coords)):
        if i != idx:
            img = color_points(img, pixel_coords[i][1], pixel_coords[i][0], ms=ms, es=es)

    img = color_points(img, pixel_coords[idx][1], pixel_coords[idx][0], fc=(255, 0, 0), ms=ms, es=es)
    return img
    
def single_img_gui(img, pixel_coords):
    h, w, _ = img.shape
    labels = []
    img_copy = img.copy()
    
    for i in range(len(pixel_coords)):
        # Add points
        img_copy = plot_all_pixels(img_copy, pixel_coords, i)
        
        # Display scene image
        fig = plt.figure(figsize=(8,8))
        plt.imshow(img_copy)
        
        # Add labelling guide to the left of the main image
        y_coord = pixel_coords[i][1]
        x_coord = pixel_coords[i][0]
        img_copy2 = img.copy()
        if y_coord-1 >= 0:
            img_copy2[y_coord-1, x_coord, :] = (255, 0, 0)
        if y_coord+1 < h:
            img_copy2[y_coord+1, x_coord, :] = (255, 0, 0)
        if x_coord-1 >= 0:
            img_copy2[y_coord, x_coord-1, :] = (255, 0, 0)
        if x_coord+1 < w:
            img_copy2[y_coord, x_coord+1, :] = (255, 0, 0)
        imagebox = OffsetImage(img_copy2[max(y_coord-3,0):min(y_coord+4,h),max(x_coord-3,0):min(x_coord+4,w),:], zoom=12)
        
        vpacker_children = [TextArea("Glass            G \nNot glass      N"), imagebox]
        box = VPacker(children=vpacker_children, align="left", pad=5, sep=5)

        # display the texts on the right side of image
        anchored_box = AnchoredOffsetbox(loc="center left",
                                            child=box,
                                            pad=0.,
                                            frameon=True,
                                            bbox_to_anchor=(1.04, 0.5),
                                            bbox_transform=plt.gca().transAxes,
                                            borderpad=0.)
        anchored_box.patch.set_linewidth(2)
        anchored_box.patch.set_facecolor('white')
        anchored_box.patch.set_alpha(1)
        anchored_box.patch.set_boxstyle("round,pad=0.5, rounding_size=0.2")
        plt.gca().add_artist(anchored_box)

        # create texts for "Enter a label for the current marker"
        box1 = TextArea("Enter a label for the current marker",
                        textprops={"weight": 'bold', "size": 12})
        box2 = DrawingArea(5, 10, 0, 0)
        box2.add_artist(mpatches.Circle((5, 5), radius=5, fc=np.array((1, 0, 0)), edgecolor="k", lw=1.5))
        box = HPacker(children=[box1, box2], align="center", pad=5, sep=5)

        # anchored_box creates the text box outside of the plot
        anchored_box = AnchoredOffsetbox(loc="lower center",
                                            child=box,
                                            pad=0.,
                                            frameon=False,
                                            bbox_to_anchor=(0.5, -0.1),  # ( 0.5, -0.1)
                                            bbox_transform=plt.gca().transAxes,
                                            borderpad=0.)
        # Formate window
        plt.gca().add_artist(anchored_box)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=2)

        # Read the images thats in the "fig" object
        buf = io.BytesIO()
        fig.savefig(buf, format="jpg", dpi=80)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        im = cv2.imdecode(img_arr, 1)
        plt.close()

        # Get label for highlighted pixel
        lbl = annotate(im)
        labels.append(lbl)
    return labels

# Specify paths to the frames
frames_path = "ExampleFrames/"

# Load the pixel database
database_path = "pixel_database.csv"
pixel_database = pd.read_csv(database_path)

# Get the unlabelled pixels and frames
filtered_db = pixel_database[pixel_database['Label'].isnull()]
frames_to_label = filtered_db.drop_duplicates(['Video','Frame Number'])[['Video','Frame Number']]

for f_index, f_row in frames_to_label.iterrows():

    # Figure out the file location of this particular frame
    vid_name = f_row['Video']
    frame_num = f_row['Frame Number']
    frame_filename = str(vid_name[:-4]) + "_Frame" + str(frame_num) + ".jpg"
    frame_path = frames_path + frame_filename

    # Load frame image
    print(frame_path)
    frame_img = cv2.imread(frame_path)
    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

    # Get pixels that need to be label for this specific frame
    this_frames_pixels = filtered_db[filtered_db['Video'] == vid_name]
    this_frames_pixels = this_frames_pixels[this_frames_pixels['Frame Number'] == frame_num]
    
    # Get unlabelled pixels for this frame
    pixel_coords = []
    for p_index, p_row in this_frames_pixels.iterrows():
        pixel_coords.append((p_row['x'], p_row['y']))

    # Get labelled from user
    output_size = 384
    img = transform.resize(frame_img, (output_size, output_size), mode='constant')
    labels = single_img_gui(img, pixel_coords)

    # Save labels that have been provided by the user
    i = 0
    for p_index, p_row in this_frames_pixels.iterrows():
        pixel_database.at[p_index, 'Label'] = int(0 if labels[i] == 'n' else 1)
        i += 1

    # Save labels to the database
    pixel_database.to_csv(database_path, index=False)  


