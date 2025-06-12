import pathlib
import utils

def main():
    config = utils.load_config()
    data_dir_path = pathlib.Path(config.get('DATA'))
    model_dir_path = pathlib.Path(config.get('MODEL'))
    output_dir = pathlib.Path('Figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = utils.find_images(data_dir_path)
    print(f"Found {len(image_paths)} images.")

    model = utils.load_model(model_dir_path)

    for img_path in image_paths:
        utils.process_image(img_path, model, output_dir)

    print("\nAll images processed.")


if __name__ == "__main__":
    main()