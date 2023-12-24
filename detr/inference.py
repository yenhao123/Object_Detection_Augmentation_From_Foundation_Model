import argparse
import json
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--val_dir', default="../../dataset/cvpdl_gligen_mix/valid/", type=str)
    parser.add_argument('--model_path', default="params/inpaint/checkpoint0349.pth", type=str)
    return parser.parse_args()

args = get_args_parser()

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

finetuned_classes = [
      'creatures', 'fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'
]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def filter_bboxes_from_outputs(outputs, image_size, threshold=0.7):
  
  # keep only predictions with confidence above threshold
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold

  probas_to_keep = probas[keep]

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], image_size)
  
  return probas_to_keep, bboxes_scaled

def plot_finetuned_results(pil_img, o_path, prob=None, boxes=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
      for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
          cl = p.argmax()
          text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
    plt.savefig(o_path)

def save_json(images, val_dir, my_model, o_dir):
  test_json = dict()

  for id in range(len(images)):  
      
      img_name = images[id]
      im = Image.open(val_dir + img_name)
      image_size = im.size

      img = transform(im).unsqueeze(0)
      outputs = my_model(img)
      for threshold in [0]:
          probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs, image_size, threshold=threshold)
          
          # 存 submit.json
          labels = [p.argmax().tolist() for p in probas_to_keep]
          scores = [p[p.argmax()].tolist() for p in probas_to_keep]
          img_info = dict()
          img_info["scores"] = scores
          img_info["labels"] = labels
          img_info["boxes"] = bboxes_scaled.tolist()
          
          test_json[img_name] = img_info

          # print(img_info)
          if id == 6:
            o_path = o_dir + img_name
            plot_finetuned_results(im, o_path, probas_to_keep, bboxes_scaled)

  with open("./submit.json", "w") as fp:
      json.dump(test_json, fp, indent=4)
  
if __name__ == "__main__":
    images = {}

    test_files = os.listdir(args.val_dir)
    for id, test_file in enumerate(test_files):
        images[id]= test_file

    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=8)

    checkpoint = torch.load(args.model_path, map_location='cpu')

    model.load_state_dict(checkpoint["model"])

    o_dir = "visualize/"
    save_json(images, args.val_dir, model, o_dir)