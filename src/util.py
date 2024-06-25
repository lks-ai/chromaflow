import torch
import cv2
import numpy as np

def apply_gaussian_blur(image, blur_amount):
    # Make sure blur_amount is a positive odd integer
    if blur_amount < 1:
        blur_amount = 1
    if blur_amount % 2 == 0:
        blur_amount += 1

    # Convert the image tensor to numpy array and rearrange dimensions for OpenCV
    image_np = image.cpu().numpy().transpose((0, 2, 1, 3))
    
    # Apply Gaussian Blur to each image in the batch
    blurred_images = []
    for i in range(image_np.shape[0]):
        blurred_image = cv2.GaussianBlur(image_np[i], (blur_amount, blur_amount), 0)
        blurred_images.append(blurred_image)
    
    # Convert the numpy array back to tensor and rearrange dimensions
    blurred_image_np = np.stack(blurred_images, axis=0).transpose((0, 2, 1, 3))
    blurred_image = torch.tensor(blurred_image_np).to(image.device)
    return blurred_image

def increase_contrast(input_tensor, threshold):
    # Convert to grayscale
    grayscale = (input_tensor[..., 0] + input_tensor[..., 1] + input_tensor[..., 2]) / 3
    # Thresholding to convert to black and white
    black_white = torch.where(grayscale > threshold, torch.tensor(1.0), torch.tensor(0.0))
    # Repeat the BW channel across the RGB dimension
    return black_white.unsqueeze(-1).repeat(1, 1, 1, 3)

def calculate_frame_count(bpm, fps, beat_count):
    """
    Calculate the number of frames needed to animate a video based on BPM, FPS, and beat count.

    Parameters:
    bpm (float): Beats per minute (tempo).
    fps (int): Frames per second of the video.
    beat_count (int): Number of beats to animate.

    Returns:
    int: Frame count for the animation.
    """
    # Calculate the duration of one beat in seconds
    beat_duration = 60 / bpm

    # Calculate the total duration for the given beat count
    total_duration = beat_duration * beat_count

    # Calculate the number of frames needed
    frame_count = total_duration * fps

    # Adjust the beat count to ensure the total duration corresponds to a whole number of frames
    adjusted_beat_count = round(frame_count / fps / beat_duration) * beat_duration
    adjusted_frame_count = round(adjusted_beat_count * fps)

    return adjusted_frame_count

def downscale_image(tensor, max_side_length):
    _, height, width, _ = tensor.shape
    
    # Determine the scaling factor
    scaling_factor = min(max_side_length / height, max_side_length / width)
    
    # Calculate new dimensions making sure they're divisible by 2
    new_height = int((height * scaling_factor) // 2 * 2)
    new_width = int((width * scaling_factor) // 2 * 2)
    
    # Use torch.nn.functional.interpolate to resize the image
    resized_tensor = torch.nn.functional.interpolate(tensor.permute(0, 3, 1, 2), size=(new_height, new_width), mode='bilinear', align_corners=False)
    return resized_tensor.permute(0, 2, 3, 1)

if __name__ == "__main__":

    # Example usage:
    bpm = 155  # Beats per minute
    fps = 30   # Frames per second
    beat_count = 4  # Number of beats to animate

    frame_count = calculate_frame_count(bpm, fps, beat_count)
    print(f"Frame count: {frame_count}")
