import json
import os
import shutil
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_labelme_to_yolo(json_file, output_dir):
    try:
        json_path = Path(json_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        image_height = data['imageHeight']
        image_width = data['imageWidth']
        
        txt_path = output_path / f"{json_path.stem}.txt"
        
        with open(txt_path, 'w') as f:
            for shape in data['shapes']:
                if shape['shape_type'] != 'polygon':
                    continue  
                
                category = shape['label']
                polygon = shape['points']
                
                normalized_polygon = []
                for x, y in polygon:
                    normalized_polygon.extend([x / image_width, y / image_height])
                
                f.write(f"{category} {' '.join(map(str, normalized_polygon))}\n")
        
        image_file = data.get('imagePath', f"{json_path.stem}.jpg")
        src_image = json_path.parent / image_file
        if not src_image.exists():
            src_image = json_path.with_name(image_file)
        dst_image = output_path / src_image.name
        
        if src_image.exists():
            shutil.copy2(src_image, dst_image)
        
        return True
    except Exception as e:
        print(f"Error processing {json_file}: {str(e)}")
        return False

def create_dataset_split(input_dir, output_dir, split_ratio=(0.7, 0.2, 0.1)):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = []
    for file_name in os.listdir(input_dir):
        if not file_name.endswith('.txt'):
            continue

        image_stem = Path(file_name).stem
        image_exists = any(
            (input_dir / f"{image_stem}{extension}").exists()
            for extension in ('.jpg', '.jpeg', '.png', '.webp')
        )
        if image_exists:
            files.append(file_name)

    random.shuffle(files)
    
    train_split = int(len(files) * split_ratio[0])
    val_split = int(len(files) * (split_ratio[0] + split_ratio[1]))
    
    train_files = files[:train_split]
    val_files = files[train_split:val_split]
    test_files = files[val_split:]
    
    for split, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for file in file_list:
            txt_src = input_dir / file
            image_stem = Path(file).stem
            image_candidates = [
                input_dir / f"{image_stem}.jpg",
                input_dir / f"{image_stem}.jpeg",
                input_dir / f"{image_stem}.png",
                input_dir / f"{image_stem}.webp",
            ]
            img_src = next((candidate for candidate in image_candidates if candidate.exists()), None)
            
            txt_dst = split_dir / file
            img_dst = split_dir / f"{image_stem}.jpg"
            
            if txt_src.exists() and img_src is not None:
                shutil.copy2(txt_src, txt_dst)
                shutil.copy2(img_src, img_dst)

def process_files(json_files, output_dir):
    max_workers = os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_labelme_to_yolo, str(json_file), output_dir) 
                   for json_file in json_files]
        
        for _ in as_completed(futures):
            pass

def main():
    base_dir = Path(__file__).resolve().parent
    json_dir = base_dir / 'img'
    output_dir = base_dir / 'labelled data'
    final_dataset_dir = output_dir / 'dataset_split'

    output_dir.mkdir(parents=True, exist_ok=True)
    final_dataset_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(json_dir.glob('*.json'))
    total_files = len(json_files)
    
    print(f"Total JSON files found: {total_files}")
    
    # Process files in batches of 15000
    batch_size = 15000
    for i in range(0, total_files, batch_size):
        batch = json_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(total_files-1)//batch_size + 1}")
        process_files(batch, output_dir)

    print("Conversion completed. Starting dataset splitting...")
    create_dataset_split(output_dir, final_dataset_dir)
    print("Dataset splitting completed.")

if __name__ == "__main__":
    main()