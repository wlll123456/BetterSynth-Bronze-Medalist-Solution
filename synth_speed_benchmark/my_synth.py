from data_juicer.core.data import NestedDataset
from data_juicer.ops import OPERATORS, load_ops
from datasets import load_dataset, concatenate_datasets
from data_juicer.core.data import NestedDataset
from data_juicer.format.formatter import unify_format
import os

"""
self.cfg_process = [
            {'image_diffusion_mapper': {'hf_diffusion': '/home/shd/workspace/dj_synth_challenge/code/models/AI-ModelScope/stable-diffusion-v1-5', 'trust_remote_code': False, 'torch_dtype': 'bf16', 'revision': 'main', 'strength': 0.8, 'guidance_scale': 7.5, 'aug_num': 1, 'keep_original_sample': True, 'caption_key': None, 'hf_img2seq': '/home/shd/workspace/dj_synth_challenge/code/models/goldsj/blip2-opt-2-7b', 'text_key': 'text', 'image_key': 'images', 'audio_key': 'audios', 'video_key': 'videos', 'accelerator': None, 'num_proc': 6, 'cpu_required': 1, 'mem_required': '8GB'
                }
            },
            {'image_text_matching_filter': {'hf_blip': 'Salesforce/blip-itm-base-coco', 'trust_remote_code': False, 'min_score': 0.003, 'max_score': 1.0, 'horizontal_flip': False, 'vertical_flip': False, 'any_or_all': 'any', 'reduce_mode': 'avg', 'text_key': 'text', 'image_key': 'images', 'audio_key': 'audios', 'video_key': 'videos', 'accelerator': None, 'num_proc': 6, 'cpu_required': 1, 'mem_required': '1500MB', 'stats_export_path': None
                }
            },
            {'image_text_similarity_filter': {'hf_clip': 'openai/clip-vit-base-patch32', 'trust_remote_code': False, 'min_score': 0.1, 'max_score': 1.0, 'horizontal_flip': False, 'vertical_flip': False, 'any_or_all': 'any', 'reduce_mode': 'avg', 'text_key': 'text', 'image_key': 'images', 'audio_key': 'audios', 'video_key': 'videos', 'accelerator': None, 'num_proc': 6, 'cpu_required': 1, 'mem_required': '1500MB', 'stats_export_path': None
                }
            },
            {'image_captioning_mapper': {'hf_img2seq': 'Salesforce/blip2-opt-2.7b', 'trust_remote_code': False, 'caption_num': 1, 'keep_candidate_mode': 'random_any', 'keep_original_sample': True, 'prompt': None, 'prompt_key': None, 'text_key': 'text', 'image_key': 'images', 'audio_key': 'audios', 'video_key': 'videos', 'accelerator': None, 'num_proc': 6, 'cpu_required': 1, 'mem_required': '16GB'
                }
            },
            {'phrase_grounding_recall_filter': {'hf_owlvit': 'google/owlvit-base-patch32', 'trust_remote_code': False, 'min_recall': 0.1, 'max_recall': 1.0, 'horizontal_flip': False, 'vertical_flip': False, 'any_or_all': 'any', 'reduce_mode': 'avg', 'iou_thr': 0.5, 'large_area_ratio_thr': 0.95, 'conf_thr': 0.0, 'text_key': 'text', 'image_key': 'images', 'audio_key': 'audios', 'video_key': 'videos', 'accelerator': None, 'num_proc': 6, 'cpu_required': 1, 'mem_required': '1GB', 'stats_export_path': None
                }
            },
            {'image_aesthetics_filter': {'hf_scorer_model': 'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE', 'trust_remote_code': False, 'min_score': 0.3, 'max_score': 1.0, 'any_or_all': 'any', 'text_key': 'text', 'image_key': 'images', 'audio_key': 'audios', 'video_key': 'videos', 'accelerator': None, 'num_proc': 6, 'cpu_required': 1, 'mem_required': '1500MB', 'stats_export_path': None
                }
            },
            {'image_watermark_filter': {'hf_watermark_model': 'amrul-hzz/watermark_detector', 'trust_remote_code': False, 'prob_threshold': 0.8, 'any_or_all': 'any', 'text_key': 'text', 'image_key': 'images', 'audio_key': 'audios', 'video_key': 'videos', 'accelerator': None, 'num_proc': 6, 'cpu_required': 1, 'mem_required': '500MB', 'stats_export_path': None
                }
            }
            ]
"""
# 
# 
# stage1先做phrase_grounding_recall_filter 0.5，再做image_text_matching_filter 0.9984，stage2对得到的jsonl文件做图文相似度过滤，过滤了0.01-0.28的数据，5.9K.然后使用5.9K数据blip重新生成caption,再使用图文相似度大于0.28的条件进行过滤，
# 得到4K数据，然后将其替换到stage1里面，最后将stage1全部复制一遍，得到最终数据集
class MySynth:
    def __init__(self, config):
            
        self.cfg_process =config

        self.cfg_op_fusion = False
        self.ops = load_ops(self.cfg_process, self.cfg_op_fusion)
    
    def warper_dataset(self, dataset):
        dataset = NestedDataset(
                concatenate_datasets([ds for _, ds in dataset.items()]))
        dataset = unify_format(dataset,
                           text_keys="text")
        return dataset
        

    def process(self, dataset:NestedDataset=None):
        source_dataset = dataset
        length = len(dataset)
        stage1 = 0
        reduce_legth = 0
        idx = 0
        for op in self.ops[:-1]:
            
            dataset = dataset.process(op)
            reduce_legth = length - len(dataset)
            length = len(dataset)
            if idx < 2 :
                stage1 =reduce_legth
                idx += 1
        # mapper op
        return source_dataset.select(range(stage1+reduce_legth +stage1+reduce_legth)).process(self.ops[-1])
        