import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xml2yolo_convert.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def normalize_class_name(cls_name):
    """Unify class names: fix typos + replace space with underscore"""
    cls_name = cls_name.strip().lower()
    # Fix typo: "feright" → "freight" (compatible with both spellings)
    cls_name = cls_name.replace("feright", "freight")
    # Replace space with underscore: "freight car" → "freight_car"
    cls_name = cls_name.replace(" ", "_")
    return cls_name

def xml2yolo(xml_dir, txt_dir, class_names):
    os.makedirs(txt_dir, exist_ok=True)
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    if not xml_files:
        logging.warning(f"No XML files found in {xml_dir}")
        return

    for xml_file in tqdm(xml_files, desc=f"Converting {os.path.basename(xml_dir)}"):
        xml_path = os.path.join(xml_dir, xml_file)
        txt_path = os.path.join(txt_dir, xml_file.replace(".xml", ".txt"))
        valid_labels = []
        label_set = set()
        polygon_missing = False

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size = root.find("size")
            if size is None:
                logging.warning(f"{xml_file} missing size node, skipped")
                continue
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)
            if img_w == 0 or img_h == 0:
                logging.warning(f"{xml_file} image width/height is 0, skipped")
                continue

            for obj in root.iter("object"):
                raw_cls_name = obj.find("name").text.strip()
                cls_name = normalize_class_name(raw_cls_name)
                
                if cls_name not in class_names:
                    logging.warning(f"{xml_file} unknown class '{raw_cls_name}' (normalized to '{cls_name}'), skipped")
                    continue
                cls_id = class_names.index(cls_name)

                polygon = obj.find("polygon")
                if polygon is None:
                    if not polygon_missing:
                        logging.warning(f"{xml_file} objects missing polygon node, skipped these objects")
                        polygon_missing = True
                    continue
                
                x1 = float(polygon.find("x1").text)
                y1 = float(polygon.find("y1").text)
                x2 = float(polygon.find("x2").text)
                y2 = float(polygon.find("y2").text)
                x3 = float(polygon.find("x3").text)
                y3 = float(polygon.find("y3").text)
                x4 = float(polygon.find("x4").text)
                y4 = float(polygon.find("y4").text)

                xmin = min(x1, x2, x3, x4)
                ymin = min(y1, y2, y3, y4)
                xmax = max(x1, x2, x3, x4)
                ymax = max(y1, y2, y3, y4)

                if xmin >= xmax or ymin >= ymax:
                    continue

                cx = (xmin + xmax) / 2 / img_w
                cy = (ymin + ymax) / 2 / img_h
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h

                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                if w < 1e-6 or h < 1e-6:
                    continue

                label_line = f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                if label_line not in label_set:
                    label_set.add(label_line)
                    valid_labels.append(label_line)

            with open(txt_path, "w", encoding="utf-8") as f:
                if valid_labels:
                    f.write('\n'.join(valid_labels))

        except Exception as e:
            logging.error(f"Convert {xml_file} failed: {str(e)}", exc_info=True)
            continue

if __name__ == "__main__":
    #DATA_ROOT = "./data/datasets/ATR-UMOD"
    DATA_ROOT = "./data/datasets/DroneVehicle"
    #CLASS_NAMES = ["car","suv","van","bus","freight_car","truck","motorcycle","trailer","excavator","crane","tank_truck"]
    CLASS_NAMES = ["car","truck","bus","van","freight_car"]
    

    convert_pairs = [
        (f"{DATA_ROOT}/rgb/labels/train", f"{DATA_ROOT}/rgb/labels/train"),
        (f"{DATA_ROOT}/rgb/labels/test", f"{DATA_ROOT}/rgb/labels/test"),
        (f"{DATA_ROOT}/rgb/labels/val", f"{DATA_ROOT}/rgb/labels/val"),
        (f"{DATA_ROOT}/ir/labels/train", f"{DATA_ROOT}/ir/labels/train"),
        (f"{DATA_ROOT}/ir/labels/test", f"{DATA_ROOT}/ir/labels/test"),
        (f"{DATA_ROOT}/ir/labels/val", f"{DATA_ROOT}/ir/labels/val")

    ]

    for xml_dir, txt_dir in convert_pairs:
        xml2yolo(xml_dir, txt_dir, CLASS_NAMES)
    
    logging.info("All directories conversion completed!")
    print("Conversion completed.")