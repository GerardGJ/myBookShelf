from PIL import Image
from ultralytics import YOLO

# This class will be used to detect the different books
# there are in the image
class BookDetector():
    def __init__(self):
        self.model = YOLO("models/yolo26n.pt")
    
    def detect_books(self,img_path:str) -> None:
        
        img = Image.open(img_path)

        results = self.model(img, conf=0.3)

        for result in results:
            boxes = result.boxes
            bounding = boxes.xyxy
            for i, bx in enumerate(bounding):
                x1, y1, x2, y2 = bx.tolist()
                cropped = img.crop((x1, y1, x2, y2))
                cropped.save(f"data/OD/bbox_{i}.jpg")

if __name__ == "__main__":
    bd = BookDetector()
    bd.detect_books("data/input/IMG_20260116_211626113.jpg")