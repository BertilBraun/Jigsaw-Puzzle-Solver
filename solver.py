import os
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

def segment_puzzle(image: IMG, puzzle_dims: DIMS) -> tuple[list[IMG], list[POINT]]:
    # Function to segment the puzzle into individual pieces
    # Implement logic to divide the puzzle image into individual pieces
    pieces, locations = [], []
    
    length, width = image.width, image.height
    puzzle_length, puzzle_width = puzzle_dims
    piece_length, piece_width = length // puzzle_length, width // puzzle_width
    
    for i in range(puzzle_length):
        for j in range(puzzle_width):
            piece = image.crop((i*piece_length, j*piece_width, (i+1)*piece_length, (j+1)*piece_width))
            pieces.append(piece)
            locations.append((i, j))
    
    return (pieces, locations)


def crop_piece(image: IMG) -> IMG:
    # Function to crop the puzzle piece from the image
    # Implement logic to crop the puzzle piece from the image
    return image # TODO implement

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

def determine_best_fits(similarity_scores: list[SCORE_INDEX], locations: list[POINT], num_best_fits=5)-> list[SCORE_POINT]:
    # Function to determine the best fits
    best_fits = []
    for score, index in similarity_scores:
        best_fits.append((score, locations[index]))
    
    best_fits.sort(reverse=True)
    return best_fits[:num_best_fits]


# Function to interpolate between green and red based on index
def index_to_color(index, total):
    # Normalize the index
    ratio = index / total
    red = int(ratio * 255)
    green = int((1 - ratio) * 255)
    return (red, green, 0)  # RGB color


def display_results(puzzle_image: IMG, puzzle_dims: DIMS, best_fit_locations: list[SCORE_POINT]) -> None:
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
    display_image.show()
    
    


def puzzle_solver(puzzle_image_path: str, puzzle_dims: DIMS):
    # Main workflow
    # Load the puzzle image and dimensions
    puzzle_image = load_image(puzzle_image_path)

    # Segment the puzzle into individual pieces
    segmented_pieces, locations = segment_puzzle(puzzle_image, puzzle_dims)

    # Encode the images
    encoded_puzzle_pieces = encode_images(segmented_pieces)
    
    while True:
        while True:
            try:
                piece_image_path = input("Enter path to puzzle piece image: ")
                
                piece_image = load_image(piece_image_path)
                break
            except FileNotFoundError:
                print("Invalid file path")
                
        cropped_piece_image = crop_piece(piece_image)
        encoded_piece = encode_images([cropped_piece_image])
        
        # Calculate similarity scores for each piece
        similarity_scores = calculate_similarity(encoded_piece, encoded_puzzle_pieces)

        # Determine the best fits
        best_fit_locations = determine_best_fits(similarity_scores, locations, num_best_fits=5)

        # Display results
        display_results(puzzle_image, puzzle_dims, best_fit_locations)


# Example usage
puzzle_solver("puzzle2.png", (6, 8))
