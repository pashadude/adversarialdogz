import json

import base64
from imageio import imwrite
from pretrainedmodels import *
from torchvision.models import *
from attack.predict import AttackPredict
from attack.whitebox import WhiteBoxAttack
from attack.blackbox import BlackBoxAttack
import requests
from PIL import Image as Im

IMAGE_DIMS = (224,224)


def req(iname):
    #entry = transforms.ToPILImage()(perturbed_image[0].squeeze_(0)).convert("RGB")
    entry = Im.open(iname)
    entry = entry.resize(IMAGE_DIMS)
    img_bytes = entry.tobytes()
    b64_img_bytes = base64.urlsafe_b64encode(img_bytes)

    try:
        r = requests.post("https://hotterdog.fly.dev/score", data=json.dumps({"input": b64_img_bytes.decode()}))
        print(r.text)
    except requests.exceptions.ConnectionError:
        print("Connection problems. Contact the CTF organizers.")
    return


# ======================================================================================

# Preparation
print('-' * 75)
result_str = 'Prediction of {} image:\n[Label:{}]-[Class:{}]-[Confidence:{:.6f}]'

# Load label-class pairs of ImageNet
class_label_dict = json.load(open('./data/imagenet_class_index.json'))
class_label = [class_label_dict[str(k)][1] for k in range(len(class_label_dict))]

# Source image
src_image_path = './data/chester.png'  # label:762
# src_image_path = './data/central_perk_299.png'  # label:762
print('Source image: [{}]'.format(src_image_path))

# Model to be attacked
model, input_size = resnet18(pretrained=True), 224
# model, input_size = resnet34(pretrained=True), 224
# model, input_size = inception_v3(pretrained=True), 299
print('Model to be attacked: [pretrained ResNet18 on ImageNet]')
print('-' * 75)

# --------------------------------------------------------------------------------------

# Prediction of source image
predictor = AttackPredict(
    model=model, input_size=input_size,
    class_label=class_label, use_cuda=True
)
src_label, src_prob, src_class = predictor.run(src_image_path)
print(result_str.format('source', src_label, src_class, src_prob))
print('-' * 75)

# ======================================================================================

# White-Box Adversarial Attack on source image
whitebox_attack = WhiteBoxAttack(
    model=model, input_size=input_size, epsilon=16, alpha=5,
    num_iters=100, early_stopping=5, use_cuda=False
)

# 'model' also could be a list of model instances
# whitebox_attack = WhiteBoxAttack(
#     model=[resnet18(pretrained=True), resnet34(pretrained=True)],
#     input_size=input_size, epsilon=16, alpha=5,
#     num_iters=100, early_stopping=5, use_cuda=True
# )

# --------------------------------------------------------------------------------------

# Non-Targeted Attack
print('White-Box Non-Targeted Adversarial Attack')
wb_nt_image = whitebox_attack(image_path=src_image_path, label=762, target=False)
wb_nt_image_path = './data/wbnjester.png'
imwrite(wb_nt_image_path, wb_nt_image)
req(wb_nt_image_path)

wb_nt_label, wb_nt_prob, wb_nt_class = predictor.run(wb_nt_image_path)
print(result_str.format('adversarial', wb_nt_label, wb_nt_class, wb_nt_prob), '\n')

# --------------------------------------------------------------------------------------

# Targeted Attack
print('White-Box Targeted Adversarial Attack')
wb_t_image = whitebox_attack(image_path=src_image_path, label=934, target=True)
wb_t_image_path = './data/wbjester.png'
imwrite(wb_t_image_path, wb_t_image)
req(wb_t_image_path)

wb_t_label, wb_t_prob, wb_t_class = predictor.run(wb_t_image_path)
print(result_str.format('adversarial', wb_t_label, wb_t_class, wb_t_prob))
print('-' * 75)

# ======================================================================================

# Black-Box Adversarial Attack on source image
blackbox_attack = BlackBoxAttack(
    model=model, input_size=input_size, epsilon=16,
    num_iters=10000, early_stopping=False, use_cuda=False, random_state=42
)

# 'model' also could be a list of model instances
# blackbox_attack = BlackBoxAttack(
#     model=[resnet18(pretrained=True), resnet34(pretrained=True)],
#     input_size=input_size, epsilon=16, num_iters=30000,
#     early_stopping=False, use_cuda=True, random_state=42
# )

# Non-Targeted Attack
print('Black-Box Non-Targeted Adversarial Attack')
bb_nt_image = blackbox_attack(src_image_path, label=934, target=False)
bb_nt_image_path = './data/bbnjester.png'
imwrite(bb_nt_image_path, bb_nt_image)
req(bb_nt_image_path)

bb_nt_label, bb_nt_prob, bb_nt_class = predictor.run(bb_nt_image_path)
print(result_str.format('adversarial', bb_nt_label, bb_nt_class, bb_nt_prob), '\n')

# --------------------------------------------------------------------------------------

# Black-Box Adversarial Attack on source image
blackbox_attack = BlackBoxAttack(
    model=model, input_size=input_size, epsilon=16,
    num_iters=15000, early_stopping=False, use_cuda=False, random_state=42
)

# Targeted Attack
print('Black-Box Targeted Adversarial Attack')
bb_t_image = blackbox_attack(src_image_path, label=934, target=True)
bb_t_image_path = './data/bbjester.png'
imwrite(bb_t_image_path, bb_t_image)
req(bb_t_image_path)

bb_t_label, bb_t_prob, bb_t_class = predictor.run(bb_t_image_path)
print(result_str.format('adversarial', bb_t_label, bb_t_class, bb_t_prob))
print('-' * 75)
