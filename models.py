def load_qwen25vl_model(model_path):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    # Qwen2.5-VL-specific imports
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return {"model": model, "processor": processor}

def run_qwen25vl(question, image_path, kwargs, temperature=0.1, top_k=50, top_p=0.9, **generation_kwargs):
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    import torch

    model = kwargs["model"]
    processor = kwargs["processor"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generation_config = {
        "max_new_tokens": 128,
        "do_sample": True if temperature > 0 else False,
        "temperature": temperature if temperature > 0 else None,
        "top_k": top_k if top_k > 0 else None,
        "top_p": top_p if top_p > 0 else None,
        **generation_kwargs
    }
    
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_config)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()

def load_robopoint_model(model_path=None):
    # Robopoint-specific imports
    from robopoint.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from robopoint.conversation import conv_templates
    from robopoint.model.builder import load_pretrained_model
    from robopoint.utils import disable_torch_init
    from robopoint.mm_utils import get_model_name_from_path

    # Disable torch initialization for faster loading
    disable_torch_init()

    # Use provided model path or default
    if model_path is None:
        model_path = 'wentao-yuan/robopoint-v1-vicuna-v1.5-13b'
    model_base = None  # Update if necessary

    # Load model name
    model_name = get_model_name_from_path(model_path)

    # Load tokenizer, model, image_processor, context_len
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

    # Prepare model kwargs
    model_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
    }

    return model_kwargs

def load_spatialvlm_model(model_path=None):
    # SpatialVLM-specific imports
    import torch
    from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration, chat_mllava

    attn_implementation = "flash_attention_2"
    if model_path is None:
        model_path = "remyxai/SpaceMantis"
    processor = MLlavaProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.float16,
        attn_implementation=attn_implementation
    )

    generation_kwargs = {
        "max_new_tokens": 1024,
        "num_beams": 1,
        "do_sample": False
    }

    model_kwargs = {
        "model": model,
        "processor": processor,
        "generation_kwargs": generation_kwargs
    }
    return model_kwargs

def load_llava_next_model(model_path=None):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    device = "cuda"
    device_map = "cuda"
    if model_path is None:
        model_path = "lmms-lab/llama3-llava-next-8b"
    model_name = "llava_llama3"

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path, None, model_name, device_map=device_map, attn_implementation=None
    )

    model.eval()
    model.tie_weights()
    model_kwargs = {"model": model, "tokenizer": tokenizer, 'image_processor': image_processor}
    return model_kwargs


def load_molmo_model(model_path=None):
    from transformers import AutoModelForCausalLM, AutoProcessor
    if model_path is None:
        model_path = "allenai/Molmo-7B-D-0924"

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='cuda'
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='cuda'
    )

    model.eval()
    model.tie_weights()
    model_kwargs = {"model": model, "processor": processor}
    return model_kwargs

def load_gpt_model(model_path=None):
    import os
    api_config = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    }
    return api_config

def load_qwen2vl_model(model_path=None):
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    if model_path is None:
        model_path = "Qwen/Qwen2-VL-7B-Instruct"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2"
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)

    model_kwargs = {
        "model": model,
        "processor": processor
    }
    return model_kwargs


def run_robopoint(question, image_path, kwargs, temperature=0.2, top_k=50, top_p=0.9, **generation_kwargs):
    # Robopoint-specific imports
    import torch
    from PIL import Image
    from robopoint.constants import (
        IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    )
    from robopoint.conversation import conv_templates
    from robopoint.mm_utils import tokenizer_image_token, process_images

    # Extract necessary components from kwargs
    model = kwargs["model"]
    tokenizer = kwargs["tokenizer"]
    image_processor = kwargs["image_processor"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Process the question
    if DEFAULT_IMAGE_TOKEN not in question:
        if getattr(model.config, 'mm_use_im_start_end', False):
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
    else:
        question = question.split('\n', 1)[1]

    # Conversation mode
    conv_mode = "llava_v1"  # Update if necessary
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize input
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).half().to(device)

    generation_config = {
        "do_sample": True if temperature > 0 else False,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p if top_p > 0 else None,
        "num_beams": 1,
        "max_new_tokens": 1024,
        "use_cache": True,
        **generation_kwargs
    }
    
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    # Generate output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            **generation_config
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def run_spatialvlm(question, image_path, kwargs, temperature=0.0, top_k=50, top_p=0.9, **generation_kwargs):
    # SpatialVLM-specific imports
    from PIL import Image
    from mantis.models.mllava import chat_mllava

    model = kwargs["model"]
    processor = kwargs["processor"]
    generation_kwargs_orig = kwargs["generation_kwargs"]
    
    updated_generation_kwargs = generation_kwargs_orig.copy()
    if temperature > 0:
        updated_generation_kwargs["do_sample"] = True
        updated_generation_kwargs["temperature"] = temperature
    if top_k > 0:
        updated_generation_kwargs["top_k"] = top_k
    if top_p > 0:
        updated_generation_kwargs["top_p"] = top_p
    
    updated_generation_kwargs.update(generation_kwargs)

    # Load the image
    image = Image.open(image_path).convert("RGB")
    images = [image]

    # Run the inference
    response, history = chat_mllava(question, images, model, processor, **updated_generation_kwargs)
    return response.strip()


def run_llava_next(question, image_path, kwargs, temperature=0.0, top_k=50, top_p=0.9, **generation_kwargs):
    # LLAVA-specific imports
    from PIL import Image
    import torch
    import copy
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    image_processor = kwargs["image_processor"]
    model = kwargs["model"]
    tokenizer = kwargs["tokenizer"]
    device = "cuda"

    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "llava_llama_3"  # Use the correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + f"\n{question}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.tokenizer = tokenizer
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    generation_config = {
        "do_sample": True if temperature > 0 else False,
        "temperature": temperature if temperature > 0 else None,
        "top_k": top_k if top_k > 0 else None,
        "top_p": top_p if top_p > 0 else None,
        "max_new_tokens": 256,
        "pad_token_id": tokenizer.eos_token_id,
        **generation_kwargs
    }
    
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        **generation_config
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs[0].strip()


def run_molmo(question, image_path, kwargs, temperature=0.0, top_k=50, top_p=0.9, **generation_kwargs):
    """
    Run the Molmo model using the generate_answer function.
    """
    def generate_answer(image_path, question, model, processor, **kwargs):
        from PIL import Image
        from transformers import GenerationConfig

        # Process the image and text
        inputs = processor.process(
            images=[Image.open(image_path)],
            text=question
        )

        # Move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # Create a GenerationConfig and update it with any additional kwargs
        generation_config = GenerationConfig(
            max_new_tokens=200, 
            stop_strings="<|endoftext|>",
            do_sample=True if kwargs.get("temperature", 0) > 0 else False
        )
        
        if "temperature" in kwargs and kwargs["temperature"] > 0:
            generation_config.temperature = kwargs["temperature"]
        if "top_k" in kwargs and kwargs["top_k"] > 0:
            generation_config.top_k = kwargs["top_k"]
        if "top_p" in kwargs and kwargs["top_p"] > 0:
            generation_config.top_p = kwargs["top_p"]
            
        for key, value in kwargs.items():
            if key not in ["temperature", "top_k", "top_p"] and hasattr(generation_config, key):
                setattr(generation_config, key, value)

        # Generate output
        output = model.generate_from_batch(
            inputs,
            generation_config,
            tokenizer=processor.tokenizer
        )

        # Extract generated tokens and decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text

    model = kwargs["model"]
    processor = kwargs["processor"]

    molmo_generation_kwargs = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        **generation_kwargs
    }
    generated_text = generate_answer(image_path, question, model, processor, **molmo_generation_kwargs)
    return generated_text


def send_question_to_openai(question, image_base64, api_config=None, temperature=0.1, top_k=50, top_p=0.9, **generation_kwargs):
    """
    Send a question and base64 encoded image to the API model and get the response.
    """
    from openai import OpenAI
    import os
    
    if api_config is None:
        api_config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        }
    
    client = OpenAI(
        api_key=api_config["api_key"],
        base_url=api_config["base_url"]
    )
    
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        generation_params = {
            "model": api_config["model_name"],
            "messages": messages,
            "max_tokens": generation_kwargs.get("max_tokens", 1024*16),
            "temperature": temperature,
        }
        
        if "top_p" in generation_kwargs:
            generation_params["top_p"] = generation_kwargs["top_p"]
        elif top_p > 0 and top_p < 1:
            generation_params["top_p"] = top_p
            
        if temperature == 0:
            generation_params["temperature"] = 0
            
            if "top_p" in generation_params:
                del generation_params["top_p"]
            
        response = client.chat.completions.create(**generation_params)
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return f"API Failed: {str(e)}"


def run_qwen2vl(question, image_path, kwargs, temperature=0.1, top_k=50, top_p=0.9, **generation_kwargs):
    """
    Use the Qwen2-VL model to answer the question about the given image.
    We'll build a 'messages' format, apply the chat template, and then generate.
    """
    import torch
    from PIL import Image

    model = kwargs["model"]
    processor = kwargs["processor"]

    device = "cuda"  # or use model.device

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    pil_image = Image.open(image_path).convert("RGB")
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt", padding=True).to(device)

    generation_config = {
        "max_new_tokens": 128,
        "do_sample": True if temperature > 0 else False,
        "temperature": temperature if temperature > 0 else None,
        "top_k": top_k if top_k > 0 else None,
        "top_p": top_p if top_p > 0 else None,
        **generation_kwargs
    }
    
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_config)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()