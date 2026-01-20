from CRAFT.main import load_craft_model, detect_text
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class WordDetector():
    def __init__(self):
        pass
    
    def modelsgetter(self):
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    def wordDetection(self,image_path:str):
        net, refine_net = load_craft_model('models/craft_mlt_25k.pth', cuda=False)
        bboxes, polys, score_text = detect_text(
            image_path, 
            net,
            result_folder='./results_test/',
            cuda=False
        )
        return bboxes
    
    def textRecognition(self,bboxees,image_path):
        words_book = []

        img = Image.open(image_path)
        for box in bboxees:
            xcorners = [box.flatten()[i] for i in [0,2,4,6]]
            ycorners = [box.flatten()[i] for i in [1,3,5,7]]
            crop = img.crop((min(xcorners), min(ycorners), max(xcorners), max(ycorners)))

            crop = crop.rotate(90, Image.NEAREST, expand = 1)\
                       .convert("RGB")
            
            pixel_values = self.processor(crop, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            words_book.append(generated_text)

        joined_book = ' '.join(words_book)
        return joined_book
    
if __name__=="__main__":
    det = WordDetector()

    print("Getting Models..")
    det.modelsgetter()
    print("Detecting bounding boxes..")
    bbox = det.wordDetection('data/OD/bbox_0.jpg')
    print("Getting the words..")
    words = det.textRecognition(bbox)
    print(words)