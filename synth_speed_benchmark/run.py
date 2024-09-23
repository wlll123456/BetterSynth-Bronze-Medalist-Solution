import os
import fire
import time
from datasets import load_dataset, concatenate_datasets
from data_juicer.core.data import NestedDataset
from data_juicer.format.formatter import unify_format

from my_synth import MySynth

# os.environ["HF-ENDPOINT"] = "https://hf-mirror.com"

def main():
    # prepare args
    dataset = load_dataset('json', data_files='test.jsonl')
    # dataset = load_dataset('json', data_files='test.jsonl', split='train')
    # print(dataset)
    # # start benchmark
    config_0 = [

            {'phrase_grounding_recall_filter': {'hf_owlvit': 'google/owlvit-base-patch32', 'trust_remote_code': False,
         'min_recall': 0.5, 'max_recall': 1.0, 'horizontal_flip': False, 'vertical_flip': False, 'any_or_all': 'any', 
         'reduce_mode': 'avg', 'iou_thr': 0.5, 'larges_area_ratio_thr': 0.95, 'conf_thr': 0.0, 'text_key': 'text', 
         'image_key': 'images', 'audio_key': 'audios', 'video_key': 'videos', 'accelerator': None, 'num_proc': 40, 
         'cpu_required': 1, 'mem_required': '1GB', 'stats_export_path': None
                }
            },
           {'image_text_matching_filter': {'hf_blip': 'Salesforce/blip-itm-base-coco', 'trust_remote_code': False,
             'torch_dtype':'bf16', 'min_score': 0.9984, 'max_score': 1.0, 'horizontal_flip': False, 'vertical_flip': False, 
             'any_or_all': 'any', 'reduce_mode': 'avg', 'text_key': 'text', 'image_key': 'images', 'audio_key': 'audios', 
             'video_key': 'videos', 'accelerator': None, 'num_proc': 15, 'cpu_required': 1, 'mem_required': '1500MB', 
             'stats_export_path': None
                }
            },
    ]
    config_1 =  [
        
            {'phrase_grounding_recall_filter': {'hf_owlvit': 'google/owlvit-base-patch32', 'trust_remote_code': False,
         'min_recall': 0.5, 'max_recall': 1.0, 'horizontal_flip': False, 'vertical_flip': False, 'any_or_all': 'any', 
         'reduce_mode': 'avg', 'iou_thr': 0.5, 'larges_area_ratio_thr': 0.95, 'conf_thr': 0.0, 'text_key': 'text', 
         'image_key': 'images', 'audio_key': 'audios', 'video_key': 'videos', 'accelerator': None, 'num_proc': 40, 
         'cpu_required': 1, 'mem_required': '1GB', 'stats_export_path': None
                }
            },
           {'image_text_matching_filter': {'hf_blip': 'Salesforce/blip-itm-base-coco', 'trust_remote_code': False,
             'torch_dtype':'bf16', 'min_score': 0.9984, 'max_score': 1.0, 'horizontal_flip': False, 'vertical_flip': False, 
             'any_or_all': 'any', 'reduce_mode': 'avg', 'text_key': 'text', 'image_key': 'images', 'audio_key': 'audios', 
             'video_key': 'videos', 'accelerator': None, 'num_proc': 15, 'cpu_required': 1, 'mem_required': '1500MB', 
             'stats_export_path': None
                }
            },
            {'image_text_similarity_filter': {'hf_clip': 'openai/clip-vit-base-patch32', 'trust_remote_code': False, 
            'torch_dtype':'bf16', 'min_score': 0.01, 'max_score': 0.28, 'horizontal_flip': False, 'vertical_flip': False, 
            'any_or_all': 'any', 'reduce_mode': 'avg', 'text_key': 'text', 'image_key': 'images', 'audio_key': 'audios', 
            'video_key': 'videos', 'accelerator': None, 'num_proc': 15, 'cpu_required': 1, 'mem_required': '1500MB', 
            'stats_export_path': None
                }
            },
            {'image_captioning_mapper': {'hf_img2seq': 'Salesforce/blip2-opt-2.7b', 'trust_remote_code': False, 
            'torch_dtype':'bf16', 'caption_num': 1, 'keep_candidate_mode': 'random_any', 'keep_original_sample': True,
             'prompt': None, 'prompt_key': None, 'text_key': 'text', 'image_key': 'images', 'audio_key': 'audios', 
             'video_key': 'videos', 'accelerator': None, 'num_proc': 2, 'cpu_required': 1, 'mem_required': '16GB'
                }
            },
            {'image_text_similarity_filter': {'hf_clip': 'openai/clip-vit-base-patch32', 'trust_remote_code': False, 
            'torch_dtype':'bf16', 'min_score': 0.28, 'max_score': 1, 'horizontal_flip': False, 'vertical_flip': False, 
            'any_or_all': 'any', 'reduce_mode': 'avg', 'text_key': 'text', 'image_key': 'images', 'audio_key': 'audios', 
            'video_key': 'videos', 'accelerator': None, 'num_proc': 15, 'cpu_required': 1, 'mem_required': '1500MB', 
            'stats_export_path': None
                }
            }
        ]
    
    runner1 = MySynth(config_1)
    # runner0 = MySynth(config_0)
    dataset = runner1.warper_dataset(dataset)

    start = time.time()
    # res0 = runner0.process(dataset)
    res1 = runner1.process(dataset)

    
    end = time.time()
    time_spend = end - start
    print(time_spend)
    return time_spend

if __name__ == '__main__':
    fire.Fire(main)