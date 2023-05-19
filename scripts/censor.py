import numpy
import torch
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image
import copy
import gradio as gr

from modules import scripts, script_callbacks, shared

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = None
safety_checker = None

def on_ui_settings():
    shared.opts.add_option("nsfw_censor_enable", shared.OptionInfo(True, "Enable NSFW Censor", section=("nsfw_censor", "NSFW Censor")))
    shared.opts.add_option("nsfw_censor_mosaic_intensity", shared.OptionInfo(64, "Mosaic intensity", gr.Slider, {"minimum": 0, "maximum": 256, "step": 1}, section=("nsfw_censor", "NSFW Censor")))

script_callbacks.on_ui_settings(on_ui_settings)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def pil_to_numpy(image):
    return (numpy.array(image.convert("RGB")) / 255.0).astype("float32")

def censor(x):
    intensity = shared.opts.nsfw_censor_mosaic_intensity
    resized = x.resize((round(x.width / intensity), round(x.height / intensity)))
    blur = resized.resize((x.width, x.height), resample=Image.NEAREST)

    return blur

# check and replace nsfw content
def check_safety(x_image):
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)

    return x_checked_image, has_nsfw_concept


def censor_batch(_x):
    x = copy.deepcopy(_x)
    x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    _x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim_numpy)

    pils = numpy_to_pil(_x.cpu().permute(0, 2, 3, 1).numpy())
    index = 0
    for h in has_nsfw_concept:
        try:
            if h is True:
                censored = pil_to_numpy(censor(pils[index]))
                _x[index] = torch.unsqueeze(torch.from_numpy(censored),0).permute(0, 3, 1, 2)
        except:
            pass
        finally:
            index += 1

    return _x

class NsfwCheckScript(scripts.Script):
    def title(self):
        return "NSFW check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        if shared.opts.nsfw_censor_enable:
            images = kwargs['images']
            images[:] = censor_batch(images)[:]
