import argparse
import os
import random
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from PIL import Image
import json
import glob
import re
random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description="idefics3 feature extraction")
parser.add_argument("--k", type=int, default=1, help="number of in-context train samples")
parser.add_argument("--outdir", type=str, default="./features")
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
args = parser.parse_args()

def index_coco_images(coco_root):
    id2path = {}
    id_pattern = re.compile(r".*_(\d{12})\.jpg$")
    for sub in ["train2014", "val2014"]:
        for p in glob.glob(os.path.join(coco_root, sub, "*.jpg")):
            m = id_pattern.match(os.path.basename(p))
            if not m:
                continue
            image_id = int(m.group(1))
            id2path[image_id] = p
    return id2path

def load_coco(json_file, coco_root, id2path=None):
    if id2path is None:
        id2path = index_coco_images(coco_root)

    samples = []
    with open(json_file, "r") as f:
        data = json.load(f)

    for item in data:
        image_id = item["image_id"]
        captions = item["caption"]

        img_path = id2path.get(image_id)
        if img_path is None:
            print(f"[WARN] Image {image_id} not found in {coco_root}")
            continue

        # 只取第一个 caption
        first_caption = captions[0] if len(captions) > 0 else ""
        samples.append({
            "image_id": image_id,
            "image_path": img_path,
            "caption": first_caption
        })

    return samples

# 使用示例
coco_root = "/datasets/coco2014"
train_json_file = "./karpathy_train_captions.json"
val_json_file = "./karpathy_val_captions.json"

train_samples= load_coco(train_json_file, coco_root)
val_samples= load_coco(val_json_file, coco_root)

print("Train:", len(train_samples), "Val:", len(val_samples))
train_samples = train_samples[:10000]
val_samples = val_samples[:5000]
print("Train:", len(train_samples), "Val:", len(val_samples))


def build_input(k_shot_samples, val_sample):

    images = []
    text_prompt = ""
    loss_mask_text = ""

    # few-shot 部分
    for s in k_shot_samples:
        images.append(Image.open(s["image_path"]).convert("RGB"))
        text_prompt += "<image>Caption: " + s["caption"] + "\n"
        loss_mask_text += "0" * len("<image>Caption: " + s["caption"] + "\n")

    images.append(Image.open(val_sample["image_path"]).convert("RGB"))
    val_prompt = "<image>Caption: "
    text_prompt += val_prompt
    loss_mask_text += "0" * len(val_prompt)  # <image>Caption:

    return images, text_prompt, loss_mask_text

def extract_features(images, text_prompt,loss_mask_text):
    inputs = processor(
        images=images,
        text=text_prompt,
        return_tensors="pt"
    ).to(model.device, torch.float16)

    #with torch.no_grad():
    #        generated_ids = model.generate(**inputs, max_new_tokens=20)
    #        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    #        print(caption)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        inputs_embeds = outputs.hidden_states[0].cpu()
        hidden_state = outputs.hidden_states[-1].cpu()

    loss_mask = torch.zeros_like(inputs["input_ids"])
    total_len = inputs["input_ids"].shape[1]
    prefix_len = len(loss_mask_text)
    if prefix_len < total_len:
        loss_mask[0, prefix_len:] = 1  # val caption token 对应1
    td = {
        "input_ids": inputs["input_ids"].cpu()[0],
        "inputs_embeds": inputs_embeds[0],
        "hidden_state": hidden_state[0],
        "loss_mask": loss_mask.cpu()[0],
    }

    # 清理显存
    del inputs, outputs, inputs_embeds, hidden_state
    torch.cuda.empty_cache()

    return td

def writedata(name, data_point, idx):
    if not os.path.exists(name):
        os.makedirs(name)
    torch.save(data_point, f"{name}/data_{idx}.ckpt")

outdir = args.outdir
if not os.path.exists(outdir):
    os.makedirs(outdir)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

model_name = "./ide3/"
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",   #
    #load_in_4bit=True,   # 4bit
    torch_dtype=torch.bfloat16,
    offload_folder="./offload",
    #llm_int8_enable_fp32_cpu_offload=True
)
processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
model.eval()


for idx, val_sample in enumerate(tqdm(val_samples, desc="Extracting features")):
    #print(val_sample)
    k_samples = random.sample(train_samples, args.k) if args.k > 0 else []
    images, text_prompt, loss_mask = build_input(k_samples, val_sample)
    outdata = extract_features(images, text_prompt,loss_mask)
    writedata(outdir, outdata, idx)