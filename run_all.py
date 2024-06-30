import json, os, tqdm
import jittor as jt

from JDiffusion.pipelines import StableDiffusionPipeline

# 设置随机数种子
# jt.misc.set_global_seed(0, different_seed_for_mpi=True)
jt.misc.set_global_seed(0)

max_num = 15
# dataset_root = "$HOME/jittor/A-Style-Figures"
dataset_root = "~/jittor/A-Style-Figures"

with jt.no_grad():
    for tempid in tqdm.tqdm(range(0, max_num)):
        taskid = "{:0>2d}".format(tempid)
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
        pipe.load_lora_weights(f"style/style_{taskid}")

        # load json
        with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
            prompts = json.load(file)

        for id, prompt in prompts.items():
            print(prompt)

            # TODO:将prompt和style连接，作为输入的prompt
            # 法1：
            image = pipe(prompt + f" in style_{taskid}", num_inference_steps=25, width=512, height=512).images[0]
            # 法2：
            # image = pipe(f"a sks ${prompt}", num_inference_steps=25, width=512, height=512).images[0]
            os.makedirs(f"./output_0629/{taskid}", exist_ok=True)
            image.save(f"./output_0629/{taskid}/{prompt}.png")
