import argparse
import json
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import time
import shortuuid
from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator
#try:
from eagle_eye.model.utils import *
from eagle_eye.model.ee_model import EeModel
from eagle_eye.model.kv_cache import initialize_past_key_values
from eagle_eye.model.choices import *
import random
import re 
import glob
from transformers.cache_utils import DynamicCache

from transformers.cache_utils import Cache

class KVCacheAdapter(Cache):
    def __init__(self, kv_list, current_length):
        """
        kv_list: ? initialize_past_key_values ??? list of [KVCache, KVCache]
        current_length: torch.Tensor ??? current_length_data
        """
        super().__init__()
        self.kv_list = kv_list
        self.current_length = current_length

    def get_seq_length(self):
        return int(self.current_length.max().item())

    def to_legacy_cache(self):
        # ?? HF ?????? list ??
        return self.kv_list

    def __len__(self):
        return len(self.kv_list)
    def __getitem__(self, idx):
        return self.kv_list[idx]

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        key_cache, value_cache = self.kv_list[layer_idx]

        # ???? key/value
        key_cache.cat(key_states)
        value_cache.cat(value_states)

        seq_len_added = key_states.shape[2]
        self.current_length[layer_idx] += seq_len_added

        return key_cache.data, value_cache.data

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

def build_input(k_shot_samples, val_sample):
    images = []
    text_prompt = ""
    loss_mask_text = ""

    for s in k_shot_samples:
        images.append(Image.open(s["image_path"]).convert("RGB"))
        text_prompt += "<image>Caption: " + s["caption"] + "\n"
        loss_mask_text += "0" * len("<image>Caption: " + s["caption"] + "\n")

    images.append(Image.open(val_sample["image_path"]).convert("RGB"))
    val_prompt = "<image>Caption: "
    text_prompt += val_prompt
    loss_mask_text += "0" * len(val_prompt)

    return images, text_prompt, loss_mask_text

def ee_forward(input_ids, pixel_values, model, tokenizer, tree_choices, logits_processor=None, max_steps=20):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place

    input_ids = input_ids.clone()
    model.ee_layer.reset_kv()
    #for name, module in model.named_modules():
    #    print(name, module.__class__.__name__)
    if hasattr(model, "tree_choices") and model.tree_choices == tree_choices:
        tree_buffers = model.tree_buffers
    else:
        tree_buffers = generate_tree_buffers(
            tree_choices,
            device=model.lm_head.weight.device,
        )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
            model.lm_head.weight.device
        )
    model.tree_buffers = tree_buffers
    model.tree_choices = tree_choices
    input_len = input_ids.shape[1]
    max_seq_len = input_len + max_steps*4

    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        current_length_data = model.current_length_data
        # Clear the length
        current_length_data.zero_()

    else:
        past_kv, past_key_values_data, current_length_data = initialize_past_key_values(
            model.language_model
        )
        past_key_values = KVCacheAdapter(past_kv, current_length_data)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data


    reset_tree_mode(model.language_model)
    print(next(model.parameters()).dtype)
    tree_logits, logits, hidden_state, sample_token = initialize_tree(
        input_ids=input_ids,
        pixel_values=pixel_values,
        model=model,
        tree_attn_mask=tree_buffers["tree_attn_mask"],
        past_key_values=past_key_values,
        logits_processor=logits_processor,
    )
    new_token = 0

    for idx in range(max_steps):
        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            tree_logits=tree_logits,
            tree_indices=tree_buffers["tree_indices"],
            retrieve_indices=tree_buffers["retrieve_indices"],
            sample_token=sample_token,
            logits_processor=logits_processor,
        )
        logits, hidden_state_new, outputs = tree_decoding(
            model=model,
            tree_candidates=tree_candidates,
            past_key_values=past_key_values,
            tree_position_ids=tree_buffers["tree_position_ids"].unsqueeze(0),
            input_ids=input_ids,
            retrieve_indices=tree_buffers["retrieve_indices_head"],
        )
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits=logits,
            candidates=candidates,
            logits_processor=logits_processor,
            cart_candidates_prob=cart_candidates_prob,
            op=tree_logits[2],
            p_indices=tree_buffers["p_indices"],
            tree_candidates=tree_candidates,
            b_indices=tree_buffers["b_indices"],
        )
        input_ids,tree_logits,new_token,hidden_state,sample_token = update_inference_inputs(
            input_ids=input_ids,
            candidates=candidates,
            best_candidate=best_candidate,
            accept_length=accept_length,
            retrieve_indices=tree_buffers["retrieve_indices"],
            logits_processor=logits_processor,
            logits=logits,
            tree_logits=tree_logits,
            new_token=new_token,
            past_key_values_data_list=past_key_values_data,
            current_length_data=current_length_data,
            model=model,
            hidden_state=hidden_state,
            hidden_state_new=hidden_state_new,
            sample_p=sample_p,
        )
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_steps:
            break

    return input_ids, new_token, idx


def run_eval(
        base_model_path,
        ee_model_path,
        model_id,
        train_samples,
        question_begin,
        question_end,
        answer_file,
        val_samples,
        datapath,
        max_new_token,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        tree_choices,
):
    data = val_samples

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(get_model_answers).remote
    else:
        get_answers_func = get_model_answers

    if use_ray:
        num_chunks = num_gpus_total // num_gpus_per_model
        chunk_size = max(1, len(data) // num_chunks)  # 防止为0
        ans_handles = []
        for i in range(0, len(data), chunk_size):
            ans_handles.append(
                get_answers_func(
                    base_model_path,
                    ee_model_path,
                    model_id,
                    train_samples,
                    answer_file,
                    data[i: i + chunk_size],
                    datapath,
                    max_new_token,
                    num_gpus_per_model,
                    max_gpu_memory,
                    temperature,
                    tree_choices,
                )
            )
        ray.get(ans_handles)
    else:
        get_answers_func(
            base_model_path,
            ee_model_path,
            model_id,
            train_samples,
            answer_file,
            data,
            datapath,
            max_new_token,
            num_gpus_per_model,
            max_gpu_memory,
            temperature,
            tree_choices,
        )


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ee_model_path,
        model_id,
        train_samples,
        answer_file,
        val_samples,
        datapath,
        max_new_token,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        tree_choices,
):
    #temperature = 0.0
    accelerator = Accelerator() 
    model = EeModel.from_pretrained(
        base_model_path=base_model_path,
        ee_model_path=ee_model_path,
        device_map="auto",  #
        load_in_4bit=True,   # 4bit
        torch_dtype=torch.bfloat16,
        offload_folder="./offload",
        llm_int8_enable_fp32_cpu_offload=True
    )
    print("after from_pretrained:", torch.cuda.memory_reserved(), torch.cuda.memory_allocated())
    state_dict = torch.load(os.path.join(ee_model_path,"model_epoch0.bin"), map_location="cpu")
    model.ee_layer.load_state_dict(state_dict, strict=False)
    
    #model = accelerator.prepare(model)
    tokenizer = model.get_tokenizer()
    #print(model.language_model.config)
    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('warmup ...')
    # warmup
    for j in range(3):
        k_samples = random.sample(train_samples, 1)
        images, text_prompt, loss_mask = build_input(k_samples, val_samples[j])
        images= Image.open("./Statue-of-Liberty-Island-New-York-Bay.jpg")
        #inputs = model.processor(images=images,text=text_prompt,return_tensors="pt")
        inputs = model.processor(text="<image>An image of",images =images , return_tensors="pt", image_grid_thw=None)

        input_ids = inputs.input_ids
        pixel_values=inputs.pixel_values
        # try:
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            output_ids, new_token, idx = ee_forward(
            input_ids=torch.as_tensor(input_ids).cuda(),
            pixel_values=torch.as_tensor(pixel_values).cuda(),
            model=model,
            tokenizer=tokenizer,
            tree_choices=tree_choices,
            logits_processor=logits_processor,
            )
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        del inputs, input_ids, pixel_values
        torch.cuda.empty_cache()
    print('Warmup done')


    torch.manual_seed(123)
    for val_sample in tqdm(val_samples):
        k_samples = random.sample(train_samples, 1)
        images, text_prompt, loss_mask = build_input(k_samples, val_sample)
        
        inputs = model.processor(images=images, text=text_prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        pixel_values = inputs.pixel_values

        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token, idx = ee_forward(
            input_ids=torch.as_tensor(input_ids).cuda(),
            pixel_values=torch.as_tensor(pixel_values).cuda(),
            model=model,
            tokenizer=tokenizer,
            tree_choices=tree_choices,
            logits_processor=logits_processor,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time

        output_ids = output_ids[0][len(input_ids[0]):]
        new_token = output_ids.shape[-1]

        output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
        if tokenizer.eos_token and tokenizer.eos_token in output:
            output = output.split(tokenizer.eos_token)[0]
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for st in special_token:
                    output = output.replace(st, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": val_sample["image_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "response": output,
                "idx": idx,
                "new_tokens": new_token,
                "wall_time": total_time,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ee-model-path",
        type=str,
        # default="/home/dhz/tmp_model/EAGLE-EYE-LLaVA-7B-10k",
        default="/seu_share/home/yangxu/230238542/ICLSD/EAGLE_EYE/eagle_eye/train/weights/ide3test",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="./ide3/",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="ide3")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="COCO-caption",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        default=0,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", 
        type=int, 
        default=0,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-dir", type=str,default="../outputs",help="The output answer file.")
    parser.add_argument("--trainjson", type=str, default="./karpathy_train_captions.json",help="Path to the input JSON file containing questions or data.")
    parser.add_argument("--valjson", type=str, default="./karpathy_val_captions.json",help="Path to the input JSON file containing questions or data.")
    parser.add_argument("--datapath", type=str, default="/seu_nvme/ogai/datasets/coco2014",help="Name or path of the dataset to be used for evaluation.")

    parser.add_argument(
        "--max-new-token",
        type=int,
        default=20,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    train_samples= load_coco(args.trainjson, args.datapath)
    val_samples= load_coco(args.valjson, args.datapath)
    train_samples = train_samples[10000:12000]
    val_samples = val_samples[4000:5000]


    args.tree_choices = eval(args.tree_choices)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    
    answer_file = os.path.join(args.answer_dir, args.bench_name, f"{args.model_id}.jsonl")
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    print(f"Output to {answer_file}")

    run_eval(
        base_model_path=args.base_model_path,
        ee_model_path=args.ee_model_path,
        model_id=args.model_id,
        train_samples=    train_samples,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        val_samples=val_samples,
        datapath = args.datapath,
        max_new_token=args.max_new_token,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        temperature=args.temperature,
        tree_choices=args.tree_choices,
    )