- Nodes:
    - N7:
        - node_type: "CLIPTextEncode"
        - text: "blurry, painting, drawing, photography, distorted, horror"
    - N5:
        - node_type: "EmptyLatentImage"
        - width: 512
        - height: 512
        - batch_size: 1
    - N9:
        - node_type: "SaveImage"
        - filename_prefix: "Result"
    - N16:
        - node_type: "ControlNetApply"
        - strength: 0.8999999999999999
    - N17:
        - node_type: "PreviewImage"
    - N8:
        - node_type: "VAEDecode"
    - N11:
        - node_type: "VAELoader"
        - vae_name: "vae-ft-mse-840000-ema-pruned.safetensors"
    - N4:
        - node_type: "CheckpointLoaderSimple"
        - ckpt_name: "dreamshaper_8.safetensors"
    - N14:
        - node_type: "ControlNetLoader"
        - control_net_name: "control_v11p_sd15_openpose_fp16.safetensors"
    - N28:
        - node_type: "DWPreprocessor"
        - detect_hand: "disable"
        - detect_body: "enable"
        - detect_face: "disable"
        - resolution: 512
        - bbox_detector: "yolox_l.onnx"
        - pose_estimator: "dw-ll_ucoco_384_bs5.torchscript.pt"
    - N6:
        - node_type: "CLIPTextEncode"
        - text: "a male, dancing in the street"
    - N24:
        - node_type: "ImageScale"
        - upscale_method: "nearest-exact"
        - width: 512
        - height: 512
        - crop: "disabled"
    - N3:
        - node_type: "KSampler"
        - seed: 249584040731174
        - control_after_generate: "randomize"
        - steps: 20
        - cfg: 7
        - sampler_name: "dpmpp_2m"
        - scheduler: "karras"
        - denoise: 1
    - N12:
        - node_type: "LoadImage"
        - image: "woman_dance.jpg"

- Links:
    - L1: N4.model -> N3.model
    - L2: N5.latent -> N3.latent_image
    - L3: N4.clip -> N6.clip
    - L5: N4.clip -> N7.clip
    - L6: N7.conditioning -> N3.negative
    - L7: N3.latent -> N8.samples
    - L9: N8.image -> N9.images
    - L12: N11.vae -> N8.vae
    - L15: N14.control_net -> N16.control_net
    - L16: N6.conditioning -> N16.conditioning
    - L17: N16.conditioning -> N3.positive
    - L43: N24.image -> N16.image
    - L45: N12.image -> N28.image
    - L46: N28.image -> N24.image
    - L47: N28.image -> N17.images
