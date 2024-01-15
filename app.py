import torch
import spaces
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, StableDiffusionXLPipeline
from transformers import AutoFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import ipown
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import gradio as gr
import cv2

base_model_path = "SG161222/RealVisXL_V3.0"
ip_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid_sdxl.bin", repo_type="model")
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
# vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    add_watermarker=False
    # vae=vae,
    #feature_extractor=safety_feature_extractor,
    #safety_checker=safety_checker
)

#pipe.load_lora_weights("h94/IP-Adapter-FaceID", weight_name="ip-adapter-faceid-plusv2_sd15_lora.safetensors")
#pipe.fuse_lora()

ip_model = ipown.IPAdapterFaceIDXL(pipe, ip_ckpt, device)

@spaces.GPU(enable_queue=True)
def generate_image(images, prompt, negative_prompt, preserve_face_structure, face_strength, likeness_strength, nfaa_negative_prompt, progress=gr.Progress(track_tqdm=True)):
    pipe.to(device)
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    faceid_all_embeds = []
    first_iteration = True
    for image in images:
        face = cv2.imread(image)
        faces = app.get(face)
        faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        faceid_all_embeds.append(faceid_embed)
        if(first_iteration and preserve_face_structure):
            face_image = face_align.norm_crop(face, landmark=faces[0].kps, image_size=224) # you can also segment the face
            first_iteration = False
            
    average_embedding = torch.mean(torch.stack(faceid_all_embeds, dim=0), dim=0)
    
    total_negative_prompt = f"{negative_prompt} {nfaa_negative_prompt}"
    
    print("Generating normal")
    image = ip_model.generate(
        prompt=prompt, negative_prompt=total_negative_prompt, faceid_embeds=average_embedding,
        scale=likeness_strength, width=1024, height=1024, guidance_scale=7.5, num_inference_steps=30
    )

    print(image)
    return image

def change_style(style):
    if style == "Photorealistic":
        return(gr.update(value=True), gr.update(value=1.3), gr.update(value=1.0))
    else:
        return(gr.update(value=True), gr.update(value=0.1), gr.update(value=0.8))

def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
css = '''
h1{margin-bottom: 0 !important}
'''
with gr.Blocks(css=css) as demo:
    gr.Markdown("# IP-Adapter-FaceID SDXL demo")
    gr.Markdown("Demo for the [h94/IP-Adapter-FaceID SDXL model](https://huggingface.co/h94/IP-Adapter-FaceID) - Non-commercial license")
    with gr.Row():
        with gr.Column():
            files = gr.Files(
                        label="Drag 1 or more photos of your face",
                        file_types=["image"]
                    )
            uploaded_files = gr.Gallery(label="Your images", visible=False, columns=5, rows=1, height=125)
            with gr.Column(visible=False) as clear_button:
                remove_and_reupload = gr.ClearButton(value="Remove and upload new ones", components=files, size="sm")
            prompt = gr.Textbox(label="Prompt",
                       info="Try something like 'a photo of a man/woman/person'",
                       placeholder="A photo of a [man/woman/person]...")
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="low quality")
            style = gr.Radio(label="Generation type", info="For stylized try prompts like 'a watercolor painting of a woman'", choices=["Photorealistic", "Stylized"], value="Photorealistic")
            submit = gr.Button("Submit")
            with gr.Accordion(open=False, label="Advanced Options"):
                preserve = gr.Checkbox(label="Preserve Face Structure", info="Higher quality, less versatility (the face structure of your first photo will be preserved). Unchecking this will use the v1 model.", value=True)
                face_strength = gr.Slider(label="Face Structure strength", info="Only applied if preserve face structure is checked", value=1.3, step=0.1, minimum=0, maximum=3)
                likeness_strength = gr.Slider(label="Face Embed strength", value=1.0, step=0.1, minimum=0, maximum=5)
                nfaa_negative_prompts = gr.Textbox(label="Appended Negative Prompts", info="Negative prompts to steer generations towards safe for all audiences outputs", value="naked, bikini, skimpy, scanty, bare skin, lingerie, swimsuit, exposed, see-through")    
        with gr.Column():
            gallery = gr.Gallery(label="Generated Images")
        style.change(fn=change_style,
                    inputs=style,
                    outputs=[preserve, face_strength, likeness_strength])
        files.upload(fn=swap_to_gallery, inputs=files, outputs=[uploaded_files, clear_button, files])
        remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files, clear_button, files])
        submit.click(fn=generate_image,
                    inputs=[files,prompt,negative_prompt,preserve, face_strength, likeness_strength, nfaa_negative_prompts],
                    outputs=gallery)
    
    # gr.Markdown("This demo includes extra features to mitigate the implicit bias of the model and prevent explicit usage of it to generate content with faces of people, including third parties, that is not safe for all audiences, including naked or semi-naked people.")
    
demo.launch()