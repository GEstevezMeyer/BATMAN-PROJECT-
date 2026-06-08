import torch 
from torchvision import transforms
from PIL import Image




def read_image(image):

    image = Image.open(image).convert("RGB")

    mean = [ 0.5600155591964722, 0.5600155591964722, 0.5600155591964722,]
    std = [ 0.41777515411376953, 0.41777515411376953, 0.41777515411376953,]
    
    transform = transforms.Compose([transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std= std)
        ])

    image = transform(image)

    return image.unsqueeze(0)


def transform_image(encoder,image,device = "cuda"):
    
    device = device if torch.cuda.is_available() else "cpu"

    encoder.to(device)
    encoder.eval()
    
    with torch.no_grad():
        image = image.to(device)
        embedding = encoder(image)

    return embedding.cpu().numpy().squeeze().tolist()



