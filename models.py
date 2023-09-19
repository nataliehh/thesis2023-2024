import transformers
import torch
import gc

class Captioning_Model():
    def __init__(self, model_type:str = 'blip', gpu = True):
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        if 'vit' == model_type:
            self.path = "nlpconnect/vit-gpt2-image-captioning"
            self.model = transformers.VisionEncoderDecoderModel.from_pretrained(self.path)
            self.processor = transformers.ViTImageProcessor.from_pretrained(self.path)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.path)
        elif 'blip' == model_type:
            self.path = "Salesforce/blip-image-captioning-large"
            self.model = transformers.BlipForConditionalGeneration.from_pretrained(self.path)
            self.processor = transformers.BlipProcessor.from_pretrained(self.path)
        elif 'git' == model_type:
            self.path = "microsoft/git-large-coco"
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.path)
            self.processor = transformers.AutoProcessor.from_pretrained(self.path)
        else:
            print('Model type not recognized.')
        try:
            self.model.to(self.device)
        except:
            print('Failed to make model GPU compatible. Running on CPU.')
            self.device = 'cpu'
            self.model.to(self.device)
    def predict(self, images, log = False): # text = None, 
        if log:
            print('Predicting...')
        if type(images) != list:
            # print('You need to provide a list of images. If you have one image, input it as: [image].')
            images = [images]
        inputs = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        gen_kwargs = {'vit': {"max_length": 16, "num_beams": 4}, 'blip': {"max_length": 20}, 'git': {"max_length": 50}}
        output = self.model.generate(pixel_values = inputs, **gen_kwargs[self.model_type])
        if 'vit' == self.model_type: 
            preds = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]
        else:
            # inputs = self.processor(images, text, return_tensors="pt").pixel_values.to(self.device)
            preds = self.processor.batch_decode(output, skip_special_tokens=True)
        if log:
            print('{} prediction(s): {}'.format(self.model_type, preds))
        del inputs
        inputs = None
        torch.cuda.empty_cache()
        gc.collect()
        return preds




