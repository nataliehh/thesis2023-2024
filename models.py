import transformers
from transformers import BlipProcessor, BlipForConditionalGeneration, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torch

class Captioning_Model():
    def __init__(self, model_type:str = 'blip'):
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'vit' == model_type:
            self.path = "nlpconnect/vit-gpt2-image-captioning"
            self.model = transformers.VisionEncoderDecoderModel.from_pretrained(self.path)
            self.processor = transformers.ViTImageProcessor.from_pretrained(self.path)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.path)
            self.model.to(self.device)
        elif 'blip' == model_type:
            self.path = "Salesforce/blip-image-captioning-large"
            self.model = transformers.BlipForConditionalGeneration.from_pretrained(self.path)
            self.processor = transformers.BlipProcessor.from_pretrained(self.path)
        elif 'git' == model_type:
            self.path = "microsoft/git-large-coco"
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.path)
            self.processor = transformers.AutoProcessor.from_pretrained(self.path)
            self.model.to(self.device)
        else:
            print('Model type not recognized.')
    def predict(self, images, text = None, log = False): 
        print('Predicting...')
        if type(images) != list:
            # print('You need to provide a list of images. If you have one image, input it as: [image].')
            images = [images]
        inputs = self.processor(images=images, return_tensors="pt")
        if 'vit' == self.model_type:
            max_length = 16
            num_beams = 4
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
            output = self.model.generate(inputs.pixel_values.to(self.device), **gen_kwargs)
            preds = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]
        elif 'blip' == self.model_type:
            inputs = self.processor(images, text, return_tensors="pt")
            output = self.model.generate(**inputs)

            preds = self.processor.batch_decode(output, skip_special_tokens=True)
        elif 'git' == self.model_type:
            gen_kwargs = {"max_length": 50}
            output = self.model.generate(pixel_values = inputs.pixel_values.to(self.device), **gen_kwargs)
            preds = self.processor.batch_decode(output, skip_special_tokens=True)
        if log:
            print('{} prediction(s): {}'.format(self.model_type, preds))
        return preds



