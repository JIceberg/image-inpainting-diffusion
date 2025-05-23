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
    fidelity_error = np.mean(original[mask == 0] - result[mask == 0]) / np.mean(result)
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

def run_diffusion(image, dt=0.1, lambda_=0.5, num_iterations=100):
    new_image = image.copy()
    for _ in range(num_iterations):
        diffusion_term = lambda_ * diffusion_iter(new_image)
        new_image += dt * diffusion_term
    return new_image

def run_diffusion_fidelity(image, mask=None, dt=0.1, lambda_=0.5, num_iterations=100, original_image=None, plot_metrics=False):
    new_image = image.copy()
    fidelity_errors = []
    inpaint_errors = []

    for iteration in range(num_iterations):
        diffusion_term = lambda_ * diffusion_iter(new_image)
        fidelity_term = (1 - lambda_) * fidelity_iter(new_image, image, mask)
        new_image += dt * (diffusion_term - fidelity_term)

        if plot_metrics and original_image is not None:
            fidelity_error, inpaint_error = compute_metrics(original_image, new_image, mask)
            fidelity_errors.append(fidelity_error)
            inpaint_errors.append(inpaint_error)

    if plot_metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(range(num_iterations), fidelity_errors, label="Fidelity Error")
        plt.plot(range(num_iterations), inpaint_errors, label="Inpaint Error")
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.title("Error Metrics vs Iteration")
        plt.legend()
        plt.grid()
        plt.show()

    return new_image

if __name__ == "__main__":
    images_dir = os.path.join(os.path.dirname(__file__), "images")
    masks_dir = os.path.join(os.path.dirname(__file__), "masks")

    image_arrays = load_images_from_directory(images_dir)
    image_filenames = [os.path.splitext(filename)[0] for filename in os.listdir(images_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    masks = load_masks_from_directory(masks_dir, image_filenames)

    print("Retrieved images:", image_filenames)

    # allowed_images = ["gojo", "statue-of-liberty", "sussy-buzz", "forest", "library"]

    images = []
    for filename, image in zip(image_filenames, image_arrays):
        # if filename not in allowed_images:
        #     continue
        matching_mask = None
        for mask_name, mask in masks.items():
            if os.path.splitext(mask_name)[0] == filename:
                matching_mask = mask
                break
        if matching_mask is not None:
            if image.shape == matching_mask.shape:
                images.append((filename, image, matching_mask))
            else:
                print(f"Warning: Shape mismatch for {filename} and its mask.")
        else:
            print(f"Warning: No mask found for {filename}.")

    for image_name, image, mask in images:
        print(f"Analyzing image {image_name}")
        image = image.astype(np.float32)
        original_image = image.copy()
        image[mask == 1] = 255
        masked_image = image.copy()

        iterations = [1000] # [10, 100, 500, 1000]
        diffused_images = []
        fidelity_images = []
        good_images = []
        for num_iterations in iterations:
            # diffused_image = run_diffusion(image, dt=0.1, lambda_=0.5, num_iterations=num_iterations)
            # diffused_images.append(diffused_image)
            # fidelity_image = run_diffusion_fidelity(image, mask=None, dt=0.1, lambda_=0.75, num_iterations=num_iterations)
            # fidelity_images.append(fidelity_image)
            good_image = run_diffusion_fidelity(image, mask=mask, dt=0.1, lambda_=0.25, num_iterations=num_iterations)
            fidelity_error, inpaint_error = compute_metrics(original_image, good_image, mask)
            print(f"Fidelity Error: {fidelity_error:.4f}, Inpaint Error: {inpaint_error:.4f}")
            good_images.append(good_image)

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(original_image, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Masked Image")
            plt.imshow(masked_image, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Diffused Image")
            plt.imshow(good_image, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        # plt.figure(figsize=(20, 15))
        # plt.suptitle('Diffusion Results', fontsize=16)

        # for i, diffused_image in enumerate(diffused_images):
        #     plt.subplot(3, 4, i + 1)
        #     plt.title(f'{iterations[i]} iterations (Diffused)')
        #     plt.imshow(diffused_image, cmap='gray')
        #     plt.axis('off')

        # for i, fidelity_image in enumerate(fidelity_images):
        #     plt.subplot(3, 4, i + 5)
        #     plt.title(f'{iterations[i]} iterations (Fidelity)')
        #     plt.imshow(fidelity_image, cmap='gray')
        #     plt.axis('off')

        # for i, good_image in enumerate(good_images):
        #     plt.subplot(3, 4, i + 9)
        #     plt.title(f'{iterations[i]} iterations (Spatially-Restricted Fidelity)')
        #     plt.imshow(good_image, cmap='gray')
        #     plt.axis('off')

        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.show()