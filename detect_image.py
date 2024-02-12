import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import requests


def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for invalid response
        image_data = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None


def calculate_image_metrics(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    laplacian_var = round(laplacian_var)

    # Calculate noise
    mean = np.mean(gray)
    stddev = np.std(gray)
    noise = stddev / mean * 100

    # Calculate contrast
    contrast = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2].std()

    # Calculate saturation
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].std()
    saturation = round(saturation)

    # Calculate SSIM score (using a reference image, or the image itself as reference)
    ssim_score = ssim(gray, gray, full=True)[0]

    # Calculate brightness
    brightness = np.mean(gray)

    # Calculate blur using the variance of Laplacian
    blur = laplacian_var > 100  # Adjust the threshold as needed

    # Assess focus
    focus = laplacian_var > 100  # Adjust the threshold as needed

    return laplacian_var, noise, contrast, saturation, ssim_score, brightness, blur, focus


def is_photo_clear(laplacian_var, noise, contrast, saturation, ssim_score, brightness, blur, focus):
    # Define thresholds for clarity assessment
    laplacian_var_threshold = list(range(200, 1001))  # Provided value
    noise_threshold = 70  # Provided value
    contrast_threshold = 70  # Provided value
    saturation_threshold = list(range(20, 100))  # Provided value
    ssim_threshold = 1.5  # Provided value
    brightness_threshold = 150  # Adjust as needed

    # Check if the image meets the clarity criteria
    if (laplacian_var in laplacian_var_threshold and
            noise < noise_threshold and
            contrast < contrast_threshold and
            saturation in saturation_threshold and
            ssim_score < ssim_threshold and
            brightness < brightness_threshold and
            blur and focus):
        return True
    else:
        return False


def process_images(image_urls):
    for idx, image_url in enumerate(image_urls):
        print(f"Processing Image {idx + 1}: {image_url}")
        image = load_image_from_url(image_url)

        if image is not None:
            laplacian_var, noise, contrast, saturation, ssim_score, brightness, blur, focus = calculate_image_metrics(
                image)

            print(f"Laplacian Variance: {laplacian_var}")
            print(f"Noise: {noise}")
            print(f"Contrast: {contrast}")
            print(f"Saturation: {saturation}")
            print(f"SSIM Score: {ssim_score}")
            print(f"Brightness: {brightness}")
            print(f"Blur: {blur}")
            print(f"Focus: {focus}")

            # Check if the image is clear
            is_clear = is_photo_clear(laplacian_var, noise, contrast, saturation, ssim_score, brightness,
                                      blur, focus)
            print("Is the image clear?", is_clear)
        else:
            print(f"Failed to load the image from URL: {image_url}")
        print()


if __name__ == "__main__":
    # URLs of the images to be processed
    image_urls = [
        "https://www.shutterstock.com/image-photo/garage-car-blurred-parking-space-260nw-1952279866.jpg",
        "https://www.shutterstock.com/image-photo/garage-car-blurred-parking-space-260nw-1938152095.jpg",
        "https://b2bchecklists.s3.amazonaws.com/drop_car_tyre_right_front-14BB917E.jpg",
        "https://b2bchecklists.s3.amazonaws.com/pickup_car_rear_seats-14BB917E.jpg"]

    # Process the images
    process_images(image_urls)
