import os
import time
import numpy as np
from tqdm import tqdm
from sd_engine import EngineOV
import cv2
from transformers import CLIPTokenizer
from diffusers.utils import logging, load_image
import PIL
from PIL import Image
import torch

import logging
logging.basicConfig(level=logging.INFO)

def get_control_net(input_shape):
    res = []
    _height, _width = input_shape
    res.append(np.zeros((2, 320, _height//8,
                _width//8)).astype(np.float32))
    res.append(np.zeros((2, 320, _height//8,
                _width//8)).astype(np.float32))
    res.append(np.zeros((2, 320, _height//8,
                _width//8)).astype(np.float32))
    res.append(np.zeros((2, 320, _height//16,
                _width//16)).astype(np.float32))
    res.append(np.zeros((2, 640, _height//16,
                _width//16)).astype(np.float32))
    res.append(np.zeros((2, 640, _height//16,
                _width//16)).astype(np.float32))
    res.append(np.zeros((2, 640, _height//32,
                _width//32)).astype(np.float32))
    res.append(np.zeros((2, 1280, _height//32,
                _width//32)).astype(np.float32))
    res.append(np.zeros((2, 1280, _height//32,
                _width//32)).astype(np.float32))
    res.append(np.zeros((2, 1280, _height//64,
                _width//64)).astype(np.float32))
    res.append(np.zeros((2, 1280, _height//64,
                _width//64)).astype(np.float32))
    res.append(np.zeros((2, 1280, _height//64,
                _width//64)).astype(np.float32))
    res.append(np.zeros((2, 1280, _height//64,
                _width//64)).astype(np.float32))
    return res

class StableDiffusionPipeline():
    def __init__(
        self,
        model_path,
        scheduler,
        dev_id = 0,
        stage = "singlize",
        controlnet_name = None,
        processor_name = None,
        tokenizer = None,
    ):
        super().__init__()

        # check configuration
        if controlnet_name != None and stage != "multilize":
            raise ValueError(f"`Controlnet cannot work in {stage} stage.")

        # controlnet
        if controlnet_name:
            controlnet_path = os.path.join(model_path, "controlnets", controlnet_name)
            self.controlnet = EngineOV(controlnet_path, device_id = dev_id)
            self.controlnet_name = controlnet_name.split("_")[0]+"_"+controlnet_name.split("_")[1]
        else:
            self.controlnet_name = None
            self.controlnet = None

        # processor net
        if processor_name:
            processor_path = os.path.join(model_path, "processors", processor_name)
            self.processor = EngineOV(processor_path, device_id = dev_id)
            if processor_name == "openpose_body_processor_fp16.bmodel":
                self.processor_hand = EngineOV(os.path.join(model_path, "processors", "openpose_hand_processor_fp16.bmodel"), device_id = dev_id)
                self.processor_face = EngineOV(os.path.join(model_path, "processors", "openpose_face_processor_fp16.bmodel"), device_id = dev_id)
        else:
            self.processor = None

        # prepare model path
        model_path = os.path.join(model_path, stage)

        if stage == "multilize":
            vae_encoder_path = os.path.join(model_path, "vae_encoder_multize.bmodel")
            vae_decoder_path = os.path.join(model_path, "vae_decoder_multize.bmodel")
            unet_path = os.path.join(model_path, "unet_multize.bmodel")
        else:
            vae_encoder_path = os.path.join(model_path, "vae_encoder_1684x_f16.bmodel")
            vae_decoder_path = os.path.join(model_path, "vae_decoder_1684x_f16.bmodel")
            unet_path = os.path.join(model_path, "unet_1684x_f16.bmodel")

        text_encoder_path = os.path.join(model_path, "text_encoder_1684x_f32.bmodel")

        # load model
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
        self.vae_encoder = EngineOV(vae_encoder_path, device_id = dev_id)
        self.vae_decoder = EngineOV(vae_decoder_path, device_id = dev_id)
        self.text_encoder = EngineOV(text_encoder_path, device_id = dev_id)
        self.unet = EngineOV(unet_path, device_id = dev_id)

        # other config
        self.device = dev_id
        self.scheduler = scheduler
        self.unet_config_in_channels = 4
        self.unet_config_sample_size = 64
        self.vae_scale_factor = 8
        self.progress_bar = None

        # use control net
        if unet_path.split('_')[-1].startswith("multize"):
            self.controlnet_flag = True
        else:
            self.controlnet_flag = False

        # init time
        self.prompt_encoder_time= 0.0
        self.vae_encoder_time = 0.0
        self.inference_time = 0.0
        self.vae_decoder_time = 0.0

    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        lora_scale = None,
    ):
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_encoder_start_time = time.time()
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            # untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            # if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            #     text_input_ids, untruncated_ids
            # ):
            #     removed_text = self.tokenizer.batch_decode(
            #         untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            #     )
            #     logger.warning(
            #         "The following part of your input was truncated because CLIP can only handle sequences up to"
            #         f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            #     )

            prompt_embeds = self.text_encoder({"tokens": np.array(text_input_ids)})
            prompt_embeds = prompt_embeds[0]

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            #uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_text_input_ids = uncond_input.input_ids
            negative_prompt_embeds = self.text_encoder({"tokens": np.array(uncond_input_text_input_ids)})

            negative_prompt_embeds = negative_prompt_embeds[0]


        if do_classifier_free_guidance:
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds], axis=0)

        self.prompt_encoder_time = time.time()-prompt_encoder_start_time

        return prompt_embeds

    def check_inputs(
        self,
        prompt,
        height,
        width,
        controlnet_img,
        init_img,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if self.controlnet and controlnet_img is None and init_img is None:
            raise ValueError(
                "`controlnet is set but there is no input for controlnet_image or init_image`"
            )

    def _encode_image(self, init_image):
        vae_encoder_start_time = time.time()
        moments = self.vae_encoder({
            "x.1": self._preprocess_image(init_image)
        })[0]

        mean, logvar = np.split(moments, 2, axis=1)
        logvar = np.clip(logvar, -30.0, 20.0)
        std = np.exp(logvar * 0.5)
        latent = (mean + std * np.random.randn(*mean.shape)) * 0.18215
        self.vae_encoder_time = time.time() - vae_encoder_start_time
        return latent

    def _prepare_canny_image(self, controlnet_img, controlnet_args={}):
        controlnet_img = np.array(controlnet_img)
        low_threshold = controlnet_args.get("low_threshold", 100)
        high_threshold = controlnet_args.get("high_threshold", 200)
        controlnet_img = cv2.Canny(controlnet_img, low_threshold, high_threshold)
        controlnet_img = controlnet_img[:, :, None]
        controlnet_img = np.concatenate([controlnet_img, controlnet_img, controlnet_img], axis=2)
        controlnet_img = Image.fromarray(controlnet_img)

        controlnet_img = controlnet_img.convert("RGB")
        controlnet_img = np.array(controlnet_img).astype(np.float32) / 255.0
        controlnet_img = [controlnet_img]
        controlnet_img = np.stack(controlnet_img, axis = 0)
        controlnet_img = controlnet_img.transpose(0, 3, 1, 2)
        controlnet_img_copy = np.copy(controlnet_img)
        controlnet_img = np.concatenate((controlnet_img,controlnet_img_copy), axis = 0)
        return controlnet_img

    def _controlnet_prepare_image(self, image):
        height, width = self.init_shape
        # opencv to PIL
        if isinstance(image, Image.Image):
            image = image
        else:
            image = Image.fromarray(image)
        image = image.resize((width, height), PIL.Image.LANCZOS) #RGB
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = image[None,:] # (1, 3, 512, 512)
        return np.concatenate((image, image), axis=0)

    def _preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w, h = image.shape[:2]
        if h != self.init_shape[0] and w != self.init_shape[1]:
            image = cv2.resize(
                image,
                (self.init_shape[1], self.init_shape[0]),
                interpolation=cv2.INTER_LANCZOS4
            )
        # normalize
        image = image.astype(np.float32) / 255.0
        image = 2.0 * image - 1.0
        image = np.expand_dims(image, 0)
        # to batch
        image = image.transpose(0, 3, 1, 2)
        return image

    def __call__(
        self,
        prompt = None,
        height = None,
        width = None,
        init_image = None,
        controlnet_img = None,
        num_inference_steps = 50,
        guidance_scale = 7.5,
        negative_prompt = None,
        num_images_per_prompt = 1,
        eta = 0.0,
        generator = None,
        latents = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        output_type = "pil",
        callback = None,
        callback_steps = 1,
        strength = 0.7,
        offset = 0,
    ):
        # 0. Default height and width to unet
        height = height or self.unet_config_sample_size * self.vae_scale_factor
        width = width or self.unet_config_sample_size * self.vae_scale_factor
        self.init_shape = [height, width]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, controlnet_img, init_image, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        if controlnet_img is None and init_image != None:
            controlnet_img = init_image

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        # text_encoder_lora_scale = (cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None)
        text_encoder_lora_scale = None
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        prompt_dtype = prompt_embeds.dtype

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet_config_in_channels
        shape = [batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor]

        if init_image is None:
            latents = np.random.randn(*shape).astype(prompt_dtype)
            timesteps = self.scheduler.timesteps + offset
        else:
            init_image = cv2.imread(init_image)
            init_image_resized = cv2.resize(init_image, (width, height))
            init_latents = self._encode_image(init_image_resized)
            init_timestep = int(num_inference_steps * strength)
            init_timestep = min(init_timestep, num_inference_steps)
            timesteps = np.array(self.scheduler.timesteps[-init_timestep:]).astype(np.long) + offset
            # noise = np.random.randn(*shape)

            # init_latents = torch.tensor(init_latents)
            # noise = torch.tensor(noise)
            # timesteps = torch.tensor(timesteps)
            # latents = self.scheduler.add_noise(init_latents, noise, timesteps[0])
            # latents = latents.numpy()
            latents = init_latents

        # 6. prepare controlnet_img
        if self.controlnet:
            if controlnet_img is not None:
                controlnet_img = load_image(controlnet_img)
                controlnet_img = controlnet_img.resize((width, height), PIL.Image.LANCZOS)
            if self.controlnet_name == "canny_controlnet":
                controlnet_img = self._prepare_canny_image(controlnet_img, {})
            elif self.controlnet_name == "openpose_controlnet":
                from openpose_utils import _prepare_openpose_image
                controlnet_img = _prepare_openpose_image(controlnet_img, self.processor, self.processor_hand, self.processor_face)
            elif self.controlnet_name == "hed_controlnet":
                from hed_utils import _prepare_hed_image
                controlnet_img = _prepare_hed_image(controlnet_img, self.processor)
            elif self.controlnet_name == "depth_controlnet":
                from depth_utils import _prepare_depth_image
                controlnet_img = _prepare_depth_image(controlnet_img, self.processor)
            elif self.controlnet_name == "segmentation_controlnet":
                from segmentation_utils import _prepare_seg_image
                controlnet_img = _prepare_seg_image(controlnet_img, self.processor)
            elif self.controlnet_name == "scribble_controlnet":
                from scribble_utils import _prepare_scribble_image
                controlnet_img = _prepare_scribble_image(controlnet_img, self.processor)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        if self.progress_bar is None:
            self.progress_bar = tqdm(total=min(num_inference_steps, len(timesteps)))

        with self.progress_bar as progress_bar:
            inference_start_time = time.time()
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
                # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                newt = np.array([t])

                # with controlnet
                if self.controlnet_flag:
                    if controlnet_img is not None:
                        controlnet_block = self.controlnet({"latent.1":latent_model_input.astype(np.float32),
                                                "t.1": newt,
                                                "prompt_embeds.1": prompt_embeds,
                                                "image.1": controlnet_img
                                                })
                    else:
                        controlnet_block = get_control_net(self.init_shape)

                    middle_block = controlnet_block[-1]
                    res_block = controlnet_block[:-1]

                    noise_pred = self.unet(
                        [latent_model_input,
                        newt,
                        prompt_embeds,
                        middle_block,
                        *res_block]
                    )[0]
                # w/o controlnet
                else:
                    noise_pred = self.unet({'latent.1':latent_model_input,
                                            't.1':newt,
                                             'prompt_embeds.1':prompt_embeds})[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # (4,64,64) () (1, 4, 64, 64)
                temp_noise_pred = torch.from_numpy(noise_pred)
                temp_latents = torch.from_numpy(latents)
                latents = self.scheduler.step(temp_noise_pred, t, temp_latents, return_dict=False)[0]
                latents = latents.numpy()

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

            self.inference_time = time.time()-inference_start_time

        #8. vae decoder
        vae_decoder_time = time.time()
        image = self.vae_decoder({"x.1": latents / 0.18215})[0]
        self.vae_decoder_time = time.time()-vae_decoder_time

        #9. postprocess
        image = (image / 2 + 0.5).clip(0, 1)
        image = (image[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
        pil_img = Image.fromarray(image)

        logging.info("prompt_encoder_time(ms): {:.2f}".format(self.prompt_encoder_time * 1000))
        logging.info("inference_time(ms): {:.2f}".format(self.inference_time * 1000))
        logging.info("vae_decoder_time(ms): {:.2f}".format(self.vae_decoder_time * 1000))

        # return with a PIL image
        return pil_img
