import torch
from transformers import pipeline

class BookAnalyzer():
    def __init__(self):
        self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
        self.pipe = self.modelGetter()

    def modelGetter(self):
        pipe = pipeline(
            "text-generation",
            model=self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        return pipe

    def book_search(self,system,book_words):
        

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": book_words},
        ]

        outputs = self.pipe(
            messages,
            max_new_tokens=256,
        )
        return outputs[0]["generated_text"][-1]['content']

if __name__=='__main__':
    book_words = "the taleb . 2 2 to black swain nassim . nicholas"

    with open("prompts/system_bookAnalyzer.md") as file:  
        data = file.read()

    ba = BookAnalyzer()
    ba.book_search(data,book_words)