import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def load_images_from_directory(directory_path):
    image_arrays = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(directory_path, filename)
            with Image.open(image_path) as img:
                img = img.convert('L')  # Convert the image to grayscale
                img_array = np.array(img)  # Convert to [H, W] matrix
                image_arrays.append(img_array)
    return image_arrays

def diffusion_iter(image):
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, laplacian_kernel)

def fidelity_iter(image, original_image, mask=None):
    fidelity_term = image - original_image
    if mask is not None:
        fidelity_term[mask == 1] = 0  # ignore mask
    # print(fidelity_term)
    return fidelity_term

def compute_metrics(original, result, mask):
    fidelity_error = np.mean(original - result) / np.mean(result)
    inpaint_error = np.mean(original[mask == 1] - result[mask == 1]) / np.mean(original[mask == 1])
    return fidelity_error, inpaint_error

def load_masks_from_directory(directory_path, image_filenames):
    masks = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) and filename.startswith('mask_'):
            image_name = os.path.splitext(filename[5:])[0]  # Remove 'mask_' prefix and file extension
            if image_name in image_filenames:
                mask_path = os.path.join(directory_path, filename)
                with Image.open(mask_path) as mask_img:
                    mask_img = mask_img.convert('L')  # Convert to grayscale
                    mask_array = np.array(mask_img)  # Convert to [H, W] matrix
                    mask_array = np.where(mask_array == 0, 1, 0)  # Black -> 1, White -> 0
                    masks[image_name] = mask_array
    return masks

if __name__ == "__main__":
    images_dir = os.path.join(os.path.dirname(__file__), "images")
    masks_dir = os.path.join(os.path.dirname(__file__), "masks")

    image_arrays = load_images_from_directory(images_dir)
    image_filenames = [os.path.splitext(filename)[0] for filename in os.listdir(images_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    masks = load_masks_from_directory(masks_dir, image_filenames)

    print(image_filenames)

    images = []
    for filename, image in zip(image_filenames, image_arrays):
        matching_mask = None
        for mask_name, mask in masks.items():
            if os.path.splitext(mask_name)[0] == filename:
                matching_mask = mask
                break
        if matching_mask is not None:
            if image.shape == matching_mask.shape:
                images.append((image, matching_mask))
            else:
                print(f"Warning: Shape mismatch for {filename} and its mask.")
        else:
            print(f"Warning: No mask found for {filename}.")

    delta_t = 0.1
    num_iterations = 1000
    lambda_ = 0.25

    for image, mask in images:
        if image.shape[0] > 1000:
            continue
        image = image.astype(np.float32)
        original_image = image.copy()
        image[mask == 1] = 255
        masked_image = image.copy()
        print(f"Processing image with shape: {image.shape}")
        for i in range(num_iterations):
            # my diffusion
            diffusion_term = lambda_ * diffusion_iter(image)
            fidelity_term = (1 - lambda_) * fidelity_iter(image, original_image, mask)
            image += delta_t * (diffusion_term - fidelity_term)
            
            # general diffusion
            # class_fidelity_term = (1 - lambda_) * fidelity_iter(class_alg_image, original_image)
            # class_diffusion_term = lambda_ * diffusion_iter(class_alg_image)
            # class_alg_image += delta_t * (class_diffusion_term - class_fidelity_term)

        fidelity_error, inpaint_error = compute_metrics(original_image, image, mask)
        print(f"Restricted Fidelity:\tFidelity Error: {fidelity_error:.4f}, Inpaint Error: {inpaint_error:.4f}")

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(original_image, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title('Masked Image')
        plt.imshow(masked_image, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title('Diffusion')
        plt.imshow(image, cmap='gray')
        plt.show()
