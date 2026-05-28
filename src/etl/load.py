from PIL import Image
from src.utils.config import create_loading_metadata
from torchvision import transforms




def read_image(image_path:str,mean,std,features_function = lambda x:x): 
    transform = transforms.Compose([transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std)
        ])

    image = Image.open(image_path).convert("RGB")
    image = features_function(image)
    image = transform(image)

    return image 

    

    

