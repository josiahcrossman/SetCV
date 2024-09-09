"""importing various modules"""
from statistics import mode

import cv2
import numpy as np

from SetLogic import Card, Game

kernel = np.ones((2, 2), np.uint8)
def order_points(pts):
    """
    Makes sure that points in rectangle are ordered properly so warping code works
    Parameters:
    pts (array): Four points determining the edges of the card

    Returns:
    rect (array): Pts ordered in clockwise fashion starting from the top left
    """
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
def get_cards(image):
    """
    Gets cards from image via filtering for the color white, looking for rectangular shapes
    Parameters:
    image: A photo of a number of set cards

    Returns:
    array: All cards shown in the input as separate images, cropped and warped
    """
    card_list = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary threshold
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    # Eroding to get rid of various artifacts
    img_erosion = cv2.erode(binary, kernel, iterations = 1)
    # Finding contours to get shapes
    contours, _ = cv2.findContours(img_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key = cv2.contourArea, reverse = True)
    for contour in (cnts):
        # Get the approximate polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # Identify the shape, contour area changes based on image resolution and camera distance
        #NOTE 7500 is good benchmark for my camera, but number might need to be tuned for different ones
        if len(approx) == 4 and cv2.contourArea(contour) > 7500:
            _, _, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            # If a non-square rectangular white object is shown, add it
            if 0.95 >= aspect_ratio and aspect_ratio <= 1.05:
                continue
        else:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rect = order_points(box)
        (tl, tr, br, bl) = rect

        # Compute the width and height of the new image for perspective transformation
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype = "float32")
        # Apply the perspective transform
        m = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, m, (max_width, max_height))

        # Calculate the center 90% region
        center_x, center_y = warped.shape[1] // 2, warped.shape[0] // 2
        crop_width, crop_height = int(warped.shape[1] * 0.9), int(warped.shape[0] * 0.9)
        start_x = center_x - crop_width // 2
        start_y = center_y - crop_height // 2

        # Crop the center 90%
        # This gets rid of any of the background that may be left on the edges of the image
        center_cropped = warped[start_y:start_y + crop_height,
                                 start_x:start_x + crop_width]
        card_list.append(center_cropped)
    return card_list
def calculate_shape(contour):
    """
    Finds the shape of the various contours within the card

    Parameters:
    contour: A contour of a shape, which is either a squiggle, oval, or diamond

    Returns:
    string: A string determining what shape the input is
    """

    # Gets the perimeter and approximates a polygon to fit that perimeter
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, .032 * peri, False)
    # Occasionally will label diamonds as having 5 sides, with the 5th side being the top/bottom
    # because of the dilation on the contour
    if len(approx) <= 5:
        return "diamond"
    # Creates a rough convex hull, if similar to contour, then it's an oval and not a squiggle
    # This is because a squiggle is not a convex hull
    hull = cv2.convexHull(approx)
    if cv2.matchShapes(contour, hull, 1, 0.0) < .08:
        return "oval"
    return "squiggle"

# NOTE The HSV values might need to be tuned based on one's environemnt.
def find_color(card):
    """
    Filter for each color, and return the one that's the most common

    Parameters:
    card (image): The cropped image of a card.

    Returns:
    string: The color the card is.

    NOTE: this will likely require some tuning depending on the lighting"""

    # Filters frame to only look at edges of the shapes, which
    # removes the white on the card that sometimes causes red to be selected
    frame = card.copy()
    blurred = cv2.blur(frame, (5,5))
    edges = cv2.Canny(blurred, 50, 75)
    img_dilation = cv2.dilate(edges, kernel, iterations = 2)
    filtered_frame = cv2.bitwise_and(frame, frame, mask = img_dilation)
    hsv = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)

    # Creating red mask
    lower_h, lower_s, lower_v = 0, 40, 0
    upper_h, upper_s, upper_v = 20, 255, 255

    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    red = cv2.countNonZero(mask)

    # Creating green mask
    lower_h, lower_s, lower_v = 50, 30, 0
    upper_h, upper_s, upper_v = 100, 255, 255

    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    green = cv2.countNonZero(mask)

    # Creating purple mask
    lower_h, lower_s, lower_v = 100, 15, 0
    upper_h, upper_s, upper_v = 150, 255, 255

    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    purple = cv2.countNonZero(mask)
    # Checks which color is most common
    if green > purple and green > red:
        return 'green'
    elif purple > red:
        return 'purple'
    else:
        return 'red'

def find_shape_info(card):
    """
    Analyzes the given card to extract shape information.

    Parameters:
    card (image): The cropped image of a card.

    Returns:
    dict: A dictionary containing shapes, number, and infill of card
    """
    frame = card.copy()
    shape_list = []
    # Blur the image and find the edges, dilate the result to make picking up contours easier
    blurred = cv2.blur(frame, (5, 5))
    edges = cv2.Canny(blurred, 50, 75)
    img_dilation = cv2.dilate(edges, kernel, iterations = 2)
    # Getting contours
    cnts, _ = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    frame_height = np.shape(frame)[0]
    # Loop through to find all the valid contours
    infill = ""
    center_pixels = []
    control_pixels = []
    shape = np.shape(frame)
    frame_height = shape[0]
    outer_pixels = frame[5 : 25, 5: 25]
    outer_pixels2 = frame[shape[0] - 25:shape[0] - 5, 
                          shape[1] - 25:shape[1] - 5]
    # The idea behind the infill detection code is to sample pixels from the inside of each of the
    # detected shapes, as well as pixels from the edges of the card, which we know are white.
    # If the pixels are drastically different, then the card must be shaded or full color inside.
    control_pixels.append(np.mean(np.mean(np.array(outer_pixels), axis = 0), axis = 0))
    control_pixels.append(np.mean(np.mean(np.array(outer_pixels2), axis = 0), axis = 0))
    try:
        for contour in cnts:
            _, _, contour_width, contour_height = cv2.boundingRect(contour)
            capture_length = round(contour_width / 6)

            # If the contour isn't greater than half the height of the card,
            # then it's not one of the shapes in the center of the card
            if contour_height / frame_height < .5:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, .03 * peri, True)
            # Sometimes other large contours picked up, but because they're typically due to the
            # card being skewed and the algorithm detecting the background on the edge of the card.
            # The resulting invalid contour will be approximated by a line/triangle. This is due to
            # the fact that one corner of the card intersects the edge of the image for a skewed 
            # card, leaving the leftover space in a triangular shape. 
            # This if statment controls for that scenario
            if len(approx) <= 3:
                continue
            shape_list.append(calculate_shape(contour))
            m = cv2.moments(contour)
            if m["m00"] != 0:
                x = int(m["m10"] / m["m00"])
                y = int(m["m01"] / m["m00"])
            surrounding_pixels = frame[y - capture_length:y + capture_length,
                                       x - capture_length:x + capture_length]
            center_pixels.append(np.mean(np.mean(np.array(surrounding_pixels), axis = 0), axis = 0))
        # Getting 1 rbg value for each image
        center_pixels = np.mean(center_pixels, axis = 0)
        control_pixels = np.mean(control_pixels, axis = 0)
        # Comparing the difference between pixels
        difference = np.subtract(center_pixels, control_pixels)
        # If the average of the rbg values at the center of the shape is < 110,
        # then there likely isn't much white, meaning that the color is solid
        if sum(center_pixels) / len(center_pixels) < 110:
            infill = 'full'
        # Compare the edges of the card with the center of the shapes,
        # if there's a large difference in rbg values, then the shapes are shaded
        elif (sum(np.absolute(difference)) / len(difference)) > 10:
            infill = 'half'
        else:
            infill = 'empty'

        # Using mode in case the model accidentally classifies a shape wrong,
        # pick the one that seems to be most common
        return str(len(shape_list)), mode(shape_list), infill, frame
    # If a card runs into an error, the code still tries to find info for the other cards and just
    # uses those, while returning the current card as invalid.
    except Exception as e:
        print(f'An error occured when finding the shape info: {e}')
        return None, None, None, frame
def create_card(card):
    """
    Gets various card attributes, checks to see if anything failed before creating a new card.
    If something failed, then the set detection algorithm only uses the cards that didn't fail. 

    Parameters:
    card (image): The cropped image of a card.

    Returns:
    Card: Card object with correct information or None if an error occured
       """
    color = find_color(card)
    number, shape, infill, frame = find_shape_info(card)
    if number is not None:
        return Card(color, number, infill, shape, frame)
    else:
        return None
def show_all_cards(images):
    """
    Shows the cards in a set. 

    Parameters:
    images (List): List of 3 card images
    """
    num_images = len(images)
    if num_images == 0:
        return
    # Find the maximum width and height of the images
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)

    # Create a blank canvas with the total grid size
    grid_image = np.zeros((max_height, 3 * max_width, 3), dtype = np.uint8)

    for idx, image in enumerate(images):
        # Calculate the row and column position
        row = idx // 3
        col = idx % 3

        # Resize image to the maximum width and height if needed
        resized_image = cv2.resize(image, (max_width, max_height))

        # Place the image on the grid
        grid_image[row * max_height: (row + 1) * max_height,
                   col * max_width: (col + 1) * max_width] = resized_image

    # Display the grid image
    cv2.imshow('Image Grid', grid_image)
    cv2.imwrite('testOutput.jpg', grid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def processing(image):
    """
    Takes an image, gets pictures of each card, finds the attributes of each card, 
    and then runs an algorithm to find all the different sets, showing each one 
    Parameters:
    image (image): A picture of a set game.
    """
    card_images = get_cards(image)
    card_list = []

    # Creating list of card objects
    for card in card_images:
        card_data = create_card(card)
        if card_data is not None:
            card_list.append(card_data)
    # Finding sets and visualizing them, or saying if none were found
    all_sets = Game.find_sets(card_list = card_list)
    if len(all_sets) == 0:
        print('NO SETS FOUND, DEAL NEW CARDS')
    for matches in all_sets:
        frame_list = []
        for card in matches:
            print(card)
            frame_list.append(card.get_image())
        show_all_cards(frame_list)
game = cv2.imread(r'setImages\testSet.jpg')
processing(game)
