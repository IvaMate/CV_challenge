import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.cluster import KMeans
from ultralytics import YOLO
from yellowbrick.cluster import KElbowVisualizer


def load_config(config_file='config.yaml'):
    '''
    Loads configuration settings from YAML file.
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_images(data_dir_path, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
    '''
    Finds all images with different extension in a directory.
    '''
    image_paths = [p for p in data_dir_path.glob('*') if p.suffix.lower() in extensions]
    return image_paths


def load_model(model_dir_path, model_file='yolov8n-seg.pt'):
    '''
    Loads a YOLOv8 segmentation model.
    '''
    print("*****Loading model...")
    return YOLO(f"{model_dir_path}/{model_file}")


def run_segmentation(model, image_rgb):
    '''
    Runs YOLO segmentation on a single image
    '''
    results = model(image_rgb)
    result = results[0]
    return result


def extract_pixels_from_masks(result, original_image_rgb):
    '''
    Extracts and combines pixels from all detected object masks in an image.
    Iterates through all masks in a segmentation result, resizes them to match
    the original image dimensions, and extracts the corresponding pixels.
    '''
    all_object_pixels = []
    for mask_data in result.masks:
        mask_tensor = mask_data.data[0]
        mask_np = mask_tensor.cpu().numpy()
        h_orig, w_orig = original_image_rgb.shape[:2]
        mask_resized = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        boolean_mask = mask_resized.astype(bool)
        object_pixels = original_image_rgb[boolean_mask]

        if object_pixels.shape[0] > 0:
            all_object_pixels.append(object_pixels)

    combined_pixels = np.vstack(all_object_pixels)
    return combined_pixels


def find_optimal_k(combined_pixels, output_dir, img_stem):
    '''
    Determines the optimal number of clusters (k) for color quantization.
    Uses the K-Elbow method on a sample of the input pixels to find the 
    optimal number of dominant colors. Saves a plot of the elbow curve to 
    the output directory.
    '''
    n_samples = min(50000, combined_pixels.shape[0])
    np.random.seed(42)
    indices = np.random.choice(combined_pixels.shape[0], size=n_samples, replace=False)
    pixel_sample = combined_pixels[indices]

    kmeans_model = KMeans(random_state=42, n_init='auto')
    visualizer = KElbowVisualizer(kmeans_model, k=(2, 10))
    visualizer.fit(pixel_sample)
    optimal_k = visualizer.elbow_value_

    elbow_filename = output_dir / f"{img_stem}_elbow_plot.png"
    visualizer.show(outpath=str(elbow_filename))
    plt.close()

    return optimal_k


def analyze_object(mask_data, original_image_rgb, result, i, optimal_k, annotated_image_rgb, img_path, output_dir):
    '''
    "Analyzes a single detected object from an image.
    '''
    mask_tensor = mask_data.data[0]
    mask_np = mask_tensor.cpu().numpy()
    h_orig, w_orig = original_image_rgb.shape[:2]
    mask_resized = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    boolean_mask = mask_resized.astype(bool)
    object_pixels = original_image_rgb[boolean_mask]

    final_model = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto').fit(object_pixels)
    palette = np.uint8(final_model.cluster_centers_)

    masked_object_viz = np.zeros_like(original_image_rgb)
    masked_object_viz[boolean_mask] = original_image_rgb[boolean_mask]

    box = result.boxes[i]
    cords = box.xyxy[0].to('cpu').numpy().astype(int)
    x1, y1, x2, y2 = cords
    cropped_viz = masked_object_viz[y1:y2, x1:x2]

    class_name = result.names[int(box.cls)]
    confidence = box.conf[0].item()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Object #{i+1}: '{class_name}' ({confidence:.2f}) - {img_path.name}", fontsize=16)

    ax1.imshow(annotated_image_rgb)
    ax1.set_title("Segmented Image")
    ax1.axis('off')

    ax2.imshow(cropped_viz)
    ax2.set_title("Masked Object")
    ax2.axis('off')

    ax3.imshow([palette])
    ax3.set_title(f"Dominant Colors (k={optimal_k})")
    ax3.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    analysis_filename = output_dir / f"{img_path.stem}_analysis_obj_{i+1}_{class_name}.png"
    plt.savefig(analysis_filename)
    plt.close(fig)
    print(f"*****Saving: {analysis_filename}")


def process_image(img_path, model, output_dir):
    '''
    
    '''
    print(f"*****Processing: {img_path.name}")
    original_image_bgr = cv2.imread(str(img_path))

    original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
    result = run_segmentation(model, original_image_rgb)

    annotated_image_rgb = result.plot()

    combined_pixels = extract_pixels_from_masks(result, original_image_rgb)

    optimal_k = find_optimal_k(combined_pixels, output_dir, img_path.stem)
    print(f"Optimal number of colors: {optimal_k}")

    for i, mask_data in enumerate(result.masks):
        analyze_object(mask_data, original_image_rgb, result, i, optimal_k, annotated_image_rgb, img_path, output_dir)
