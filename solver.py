import os
from turtle import width
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageDraw

from torch import Tensor
import torch

POINT = tuple[int, int]
DIMS = tuple[int, int]
SCORE_INDEX = tuple[int, int]
SCORE_POINT = tuple[int, POINT]
IMG = Image.Image

# Load the OpenAI CLIP Model
print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32')

def load_image(image_path:str) -> IMG:
    # Function to load image
    if not os.path.exists(image_path):
        if os.path.exists(f"{image_path}.png"):
            image_path = f"{image_path}.png"
        elif os.path.exists(f"{image_path}.jpg"):
            image_path = f"{image_path}.jpg"
        else:
            raise FileNotFoundError("Invalid file path")
    return Image.open(image_path)

def encode_images(images) -> Tensor:
    global model
    encoded = model.encode(images, batch_size=128, convert_to_tensor=True, show_progress_bar=True)
    if isinstance(encoded, Tensor):
        return encoded
    raise TypeError("Expected Tensor, got {}".format(type(encoded)))

def segment_puzzle(image: IMG, puzzle_dims: DIMS) -> list[IMG]:
    # Function to segment the puzzle into individual pieces
    # Implement logic to divide the puzzle image into individual pieces
    pieces = []
    
    length, width = image.width, image.height
    puzzle_length, puzzle_width = puzzle_dims
    piece_length, piece_width = length // puzzle_length, width // puzzle_width
    
    for i in range(puzzle_length):
        for j in range(puzzle_width):
            piece = image.crop((i*piece_length, j*piece_width, (i+1)*piece_length, (j+1)*piece_width))
            pieces.append(piece)
    
    return pieces

def get_pice_mask(image: np.ndarray) -> np.ndarray:
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray   = cv2.medianBlur(gray, ksize=5)
    thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.blur(thresh, ksize=(3, 3))
    
    ret, labels = cv2.connectedComponents(thresh)
    
    # get mask with largest area
    mask = np.array(labels, dtype=np.uint8)
    max_area_label = np.argmax(np.bincount(labels.flat)[1:]) + 1  # type: ignore
    mask[labels != max_area_label] = 0
    mask[labels == max_area_label] = 255
    
    return mask
    

def shrink_crop_image(image: np.ndarray):
    # if image is wider than tall, rotate it
    if image.shape[1] > image.shape[0]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    height, width = image.shape[:2]
    piece_mask = get_pice_mask(image)
    
    # remap piece_mask to 0 and 1
    piece_mask[piece_mask > 0] = 1
    
    # We want to remove columns and rows while the sum of the pixels in the column/row is less than a percentage of column/row pixels
    THRESH_ROW = 0.40
    THRESH_COL = 0.40
    
    # if the image is not approximately square, we need to set the threshold for columns to be .25
    if width / height > 1.1 or height / width > 1.1:
        THRESH_COL = 0.25
    
    to_remove = [1, 1, 1, 1]
    
    for i in range(width):
        if np.sum(piece_mask[:, i]) < THRESH_COL * height:
            to_remove[0] += 1
        else:
            break
        
    for i in range(width-1, 0, -1):
        if np.sum(piece_mask[:, i]) < THRESH_COL * height:
            to_remove[1] += 1
        else:
            break
        
    for i in range(height):
        if np.sum(piece_mask[i, :]) < THRESH_ROW * width:
            to_remove[2] += 1
        else:
            break
        
    for i in range(height-1, 0, -1):
        if np.sum(piece_mask[i, :]) < THRESH_ROW * width:
            to_remove[3] += 1
        else:
            break
        
    image = image[to_remove[2]:height-to_remove[3], to_remove[0]:width-to_remove[1]]
    piece_mask = piece_mask[to_remove[2]:height-to_remove[3], to_remove[0]:width-to_remove[1]]
            
    # set the background to black
    image[piece_mask == 0] = 0
    
    return image

def crop_piece(img: IMG) -> IMG:
    # Function to crop the puzzle piece from the image
    # parse cv2 image from PIL
    image = np.array(img)
    
    piece_mask = get_pice_mask(image)
    
    # Find contours
    contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour which should be the puzzle piece
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the minimal rotated rectangle that encloses the largest contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)  # type: ignore

    # Get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # Coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # The perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(image, M, (width, height))

    shrink_cropped = shrink_crop_image(warped)
    
    # parse the image back to PIL from cv2
    return Image.fromarray(shrink_cropped)

def calculate_similarity(encoded_piece: Tensor, encoded_puzzle_pieces: Tensor) -> list[SCORE_INDEX]:
    # Function to calculate similarity scores
    # Implement similarity calculation for each puzzle piece here
    
    all_encodings = torch.cat([encoded_piece, encoded_puzzle_pieces])
    
    # Now we run the clustering algorithm. This function compares images against 
    # all other images and returns a list with the pairs that have the highest 
    # cosine similarity score
    processed_images = util.paraphrase_mining_embeddings(all_encodings)
    
    similarities : list[SCORE_INDEX] = []
    for score, i, j in processed_images:
        if i == 0:
            similarities.append((score, j-1)) # Subtract 1 to account for the piece itself
        elif j == 0:
            similarities.append((score, i-1)) # Subtract 1 to account for the piece itself

    return similarities

def get_piece_location(puzzle_dims: DIMS, index: int) -> POINT:
    puzzle_length, puzzle_width = puzzle_dims

    # Calculate row and column based on index
    row = index // puzzle_width
    column = index % puzzle_width

    return (row, column)

def determine_best_fits(similarity_scores: list[SCORE_INDEX], puzzle_dims: DIMS, num_best_fits=5)-> list[SCORE_POINT]:
    # Function to determine the best fits
    best_fits = []
    for score, index in similarity_scores:
        best_fits.append((score, get_piece_location(puzzle_dims, index)))
    
    best_fits.sort(reverse=True)
    return best_fits[:num_best_fits]

# Function to interpolate between green and red based on index
def index_to_color(index, total):
    # Normalize the index
    ratio = index / total
    red = int(ratio * 255)
    green = int((1 - ratio) * 255)
    return (red, green, 0)  # RGB color

def render_results(puzzle_image: IMG, puzzle_dims: DIMS, best_fit_locations: list[SCORE_POINT]) -> Image.Image:
    # Show an image with puzzle_image as the background, an overlay of the puzzle piece locations, and filled in best fit piece locations
    # Puzzle dimensions in pieces
    puzzle_length, puzzle_width = puzzle_dims

    # Calculate the size of each piece
    piece_length, piece_width = puzzle_image.width / puzzle_length, puzzle_image.height / puzzle_width

    # Make a copy of the puzzle image to avoid modifying the original
    display_image = puzzle_image.copy()
    # Create a drawing context
    draw = ImageDraw.Draw(display_image)

    # Loop through each grid position and draw the grid
    for i in range(puzzle_length):
        for j in range(puzzle_width):
            top_left = (i * piece_length, j * piece_width)
            bottom_right = ((i + 1) * piece_length, (j + 1) * piece_width)
            draw.rectangle((top_left, bottom_right), outline="black")

    best_fit_locations.sort(reverse=True, key=lambda x: x[0])
    
    # Overlay the best fit locations
    for index, (score, (x, y)) in enumerate(best_fit_locations):
        top_left = (x * piece_length, y * piece_width)
        bottom_right = ((x + 1) * piece_length, (y + 1) * piece_width)
        # Determine the color based on the score
        color = index_to_color(index, len(best_fit_locations) - 1)
        
        draw.rectangle((top_left, bottom_right), outline=color, width=3)

        # Draw the score
        score_text = "{:.2f}".format(score)
        draw.text((top_left[0] + 10, top_left[1] + 10), score_text, fill="red")

    # Display the image
    return display_image
    
def setup_solver(puzzle_image_path: str, puzzle_dims: DIMS) -> Tensor:
    # Main workflow
    # Load the puzzle image and dimensions
    puzzle_image = load_image(puzzle_image_path)

    # Segment the puzzle into individual pieces
    segmented_pieces = segment_puzzle(puzzle_image, puzzle_dims)

    # Encode the images
    return encode_images(segmented_pieces)

def solve_piece(piece_image: IMG, puzzle_image_path: str, puzzle_dims: DIMS, encoded_puzzle_pieces: Tensor) -> tuple[IMG, IMG]:
      
    cropped_piece_image = crop_piece(piece_image)
    encoded_piece = encode_images([cropped_piece_image])
    
    # Calculate similarity scores for each piece
    similarity_scores = calculate_similarity(encoded_piece, encoded_puzzle_pieces)

    # Determine the best fits
    best_fit_locations = determine_best_fits(similarity_scores, puzzle_dims, num_best_fits=5)
    
    return render_results(load_image(puzzle_image_path), puzzle_dims, best_fit_locations), cropped_piece_image

def get_image_input() -> IMG:
    while True:
        try:
            image_path = input("Enter path to image: ")
            return load_image(image_path)
        except FileNotFoundError:
            print("Invalid file path")

def puzzle_solver(puzzle_image_path: str, puzzle_dims: DIMS):
    
    encoded_puzzle_pieces = setup_solver(puzzle_image_path, puzzle_dims)
    
    while True:
        piece_image = get_image_input()
        
        result, _ = solve_piece(piece_image, puzzle_image_path, puzzle_dims, encoded_puzzle_pieces)
        # Display results
        result.show()


if __name__ == "__main__":
    # Example usage
    # puzzle_solver("samples/puzzle2.png", (6, 8))
    crop_piece(load_image("samples/crop_piece.jpg")).show()
