from bookAnalysis import BookAnalyzer
from wordsExtraction import WordDetector
from bookDetector import BookDetector
import os

def orchestrator():
    print("0. Initiliza all classes")
    wordDetector = WordDetector()
    wordDetector.modelsgetter()
    bookDetector = BookDetector()
    bookAnalyzer = BookAnalyzer()

    print("1. Detecting the differnt books using YOLO")
    bookDetector.detect_books("data/input/IMG_20260116_211626113.jpg")
    print("All books detected succesfully")
    
    books = []
    directoy_books = 'data/OD'
    for book in os.listdir(directoy_books):
        image_path = os.path.join(directoy_books,book)
        print("2. Getting all the words form the book")
        bbox = wordDetector.wordDetection(image_path)
        print("Getting the words..")
        words = wordDetector.textRecognition(bbox,image_path)
        print("All words gotten successfully")

        print("3. Start analyzing the books to identify it")
        with open("prompts/system_bookAnalyzer.md") as file:  
            system_prompt = file.read()
        book = bookAnalyzer.book_search(system_prompt, words)
        print("The book you are looking for is:")
        books.append(book)
    for book in books:
        print(book)


orchestrator()