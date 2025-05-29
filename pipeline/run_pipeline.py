import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
import json
import os
import argparse
import mmengine # For config loading
import jsonpickle # For lm-eval harness results
import sys
import gc # For garbage collection
from tqdm import tqdm
import numpy as np # For np.mean in evaluation
from typing import List, Optional, Tuple, Callable, Any, Dict

# --- Import from your existing/modified project structure ---
from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.model_utils.model_base import ModelBase

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    add_hooks,
    get_activation_addition_input_pre_hook,
    get_all_direction_ablation_hooks,
    get_ebm_intervention_hook # Assumed to be defined in hook_utils.py
)


from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores, kl_div_fn, plot_refusal_scores, filter_fn_ablation, filter_fn_addition, get_last_position_logits
from pipeline.submodules.evaluate_jailbreak import substring_matching_judge_fn, evaluate_jailbreak
from pipeline.evaluator.evalharness import LMEvalHarness

# --- New Import for EBM ---
from pipeline.ebm.ebm_model import SimpleEBM, load_ebm_model # Assuming ebm_model.py is in pipeline/ebm/

# --- Logger Class ---
class Logger:
    def __init__(self, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(log_file, "a", buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run intervention pipeline (with optional EBM training).")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config YAML file')
    parser.add_argument('--model_path', type=str, required=False, default=None, help='Override model_path from config')
    parser.add_argument('--batch_size', type=int, required=False, default=None, help='Override batch_size from config')
    parser.add_argument('--force_retrain_ebm', action='store_true', help='Force retraining of EBMs even if they exist.')
    return parser.parse_args()

# +++ Add definitions for missing evaluation functions +++
def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None, system=None):
    """Generate and save completions for a dataset."""
    completions_dir = os.path.join(cfg.artifact_path, 'completions')
    os.makedirs(completions_dir, exist_ok=True)

    if dataset is None:
        # Assuming load_dataset_split or a similar function can handle dataset_name directly if it's a string key
        # This part might need adjustment based on how dataset_name is structured and how load_dataset is meant to be used here.
        # For now, if dataset is a string, we assume it needs to be loaded via a generic loader.
        # If dataset_name is a key to preloaded data (like in test_data_cache), 'dataset' should be passed directly.
        print(f"Loading dataset for {dataset_name} as it was not provided directly.")
        # Placeholder: You might need a more specific way to load datasets by name if they aren't in test_data_cache
        # For example, using load_dataset_split or load_dataset with appropriate harmtype/split if dataset_name implies it.
        # This is a potential point of failure if dataset_name isn't directly loadable by a generic `load_dataset` function.
        try:
            # Attempt to load as if it's a harmtype for a 'test' split by default for general eval sets
            loaded_prompts = load_dataset(dataset_name) 
            # Ensure loaded_prompts is a list of strings, which is what subsequent code expects.
            # load_dataset might return various formats, adjust as needed.
            # For now, assuming it returns a list of strings (prompts).
            # If it returns list of dicts, or a Hugging Face Dataset object, this will need adaptation.
            if not isinstance(loaded_prompts, list) or (loaded_prompts and not isinstance(loaded_prompts[0], str)):
                # Attempt to extract instructions if it's a list of dicts with 'instruction' key
                if isinstance(loaded_prompts, list) and loaded_prompts and isinstance(loaded_prompts[0], dict) and 'instruction' in loaded_prompts[0]:
                    loaded_prompts = [item['instruction'] for item in loaded_prompts]
                elif hasattr(loaded_prompts, 'map'): # Basic check for HF Dataset object
                    try:
                        # Common column names for prompts in HF datasets
                        if 'prompt' in loaded_prompts.column_names:
                            loaded_prompts = loaded_prompts['prompt']
                        elif 'text' in loaded_prompts.column_names:
                            loaded_prompts = loaded_prompts['text']
                        elif 'instruction' in loaded_prompts.column_names: # Already common
                             loaded_prompts = loaded_prompts['instruction']
                        else: # Add more common prompt column names if necessary
                            raise ValueError(f"Dataset {dataset_name} loaded via load_dataset, but could not identify prompt column.")
                    except Exception as e_hf_extract:
                        print(f"Warning: Tried to process loaded dataset {dataset_name} as HF Dataset but failed: {e_hf_extract}. Assuming direct list of prompts or erroring.")

            if not isinstance(loaded_prompts, list) or (loaded_prompts and not isinstance(loaded_prompts[0], str)):
                 raise TypeError(f"Dataset {dataset_name} loaded by load_dataset did not result in a list of prompt strings. Actual type: {type(loaded_prompts)}")

            # Perform sampling after loading if n_test is configured and dataset is larger
            if cfg.get('n_test') is not None and len(loaded_prompts) > cfg.n_test:
                loaded_prompts = random.sample(loaded_prompts, cfg.n_test)
            dataset = [{'instruction': p, 'category': dataset_name} for p in loaded_prompts]
            if not dataset:
                print(f"Warning: Loaded empty dataset for {dataset_name}. Check dataset name and loading logic.")
                # Fallback or raise error if necessary
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}. Completions might be empty or incorrect if dataset is not pre-passed.")
            dataset = [] # Ensure dataset is an empty list to avoid errors in generate_completions

    if not dataset:
        print(f"Warning: Dataset for '{dataset_name}' is empty. Skipping completion generation.")
        # Save an empty list of completions
        completions_to_save = []
    else:
        print(f"Generating completions for {dataset_name} with {len(dataset)} examples...")
        completions_to_save = model_base.generate_completions(
            dataset, 
            fwd_pre_hooks=fwd_pre_hooks, 
            fwd_hooks=fwd_hooks, 
            max_new_tokens=cfg.get('max_new_tokens', 512),
            batch_size=cfg.batch_size, 
            system=system
        )
    
    completion_save_path = os.path.join(completions_dir, f'{dataset_name}_{intervention_label}_completions.json')
    with open(completion_save_path, "w") as f:
        json.dump(completions_to_save, f, indent=4)
    print(f"Completions for {dataset_name} saved to {completion_save_path}")

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    completions_dir = os.path.join(cfg.artifact_path, 'completions')
    evaluations_dir = os.path.join(cfg.artifact_path, 'evaluations') # Changed from 'completions' to 'evaluations'
    os.makedirs(evaluations_dir, exist_ok=True)

    completion_file_path = os.path.join(completions_dir, f'{dataset_name}_{intervention_label}_completions.json')
    evaluation_save_path = os.path.join(evaluations_dir, f'{dataset_name}_{intervention_label}_evaluations.json')

    if not os.path.exists(completion_file_path):
        print(f"Error: Completions file not found at {completion_file_path}. Skipping evaluation for {dataset_name}.")
        return
        
    with open(completion_file_path, 'r') as f:
        completions = json.load(f)
    
    if not completions:
        print(f"Warning: No completions found in {completion_file_path} for {dataset_name}. Skipping evaluation.")
        # Save an empty evaluation object or handle as appropriate
        with open(evaluation_save_path, "w") as f:
            json.dump({"error": "No completions to evaluate"}, f, indent=4)
        return

    # Ensure evaluate_jailbreak is imported or defined
    # from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak # Assuming this path

    print(f"Evaluating completions for {dataset_name} using methodologies: {eval_methodologies}")
    evaluation_results = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=evaluation_save_path, # evaluate_jailbreak saves it itself
        cfg=cfg # Pass cfg if evaluate_jailbreak uses it for things like API keys
    )

    with open(evaluation_save_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Evaluation results for {dataset_name} saved to {evaluation_save_path}")

# We also need the LMEvalHarness class or a wrapper for eval_harness
# from pipeline.evaluator.evalharness import LMEvalHarness # Assuming this path

def eval_harness(cfg, model_base, identifier):
    eval_results_combined = {}
    
    # MMLU specific eval if configured
    if hasattr(cfg, 'eval_harness_mmlu') and cfg.eval_harness_mmlu:
        print(f"Running LM Eval Harness for MMLU (identifier: {identifier})...")
        eval_harness_evaluator_mmlu = LMEvalHarness(cfg.eval_harness_mmlu)
        lm_eval_results_mmlu = eval_harness_evaluator_mmlu.evaluate(
            model=model_base
        )
        eval_results_combined['mmlu'] = lm_eval_results_mmlu
        
        results_dir = os.path.join(cfg.artifact_path, 'lm_eval_results')
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f'{identifier}_mmlu_results.json'), "w") as f:
            f.write(jsonpickle.encode(lm_eval_results_mmlu, indent=4))
        print(f"MMLU LM Eval Harness results saved for {identifier}.")

    # General eval harness tasks if configured
    if hasattr(cfg, 'eval_harness') and cfg.eval_harness and cfg.eval_harness.get('tasks'):
        print(f"Running general LM Eval Harness tasks (identifier: {identifier})...")
        eval_harness_evaluator_general = LMEvalHarness(cfg.eval_harness)
        lm_eval_results_general = eval_harness_evaluator_general.evaluate(
            model=model_base
        )
        eval_results_combined['general'] = lm_eval_results_general
        
        results_dir = os.path.join(cfg.artifact_path, 'lm_eval_results') # Redundant if mmlu ran, but harmless
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f'{identifier}_general_results.json'), "w") as f:
            f.write(jsonpickle.encode(lm_eval_results_general, indent=4))
        print(f"General LM Eval Harness results saved for {identifier}.")
        
    if not eval_results_combined:
        print(f"No LM Eval Harness tasks configured or run for identifier: {identifier}.")
        return None
        
    return eval_results_combined

# --- Dataset Loading and Filtering ---
def load_and_sample_datasets(cfg):
    random.seed(cfg.random_seed)
    data1_train = random.sample(load_dataset_split(harmtype=cfg.harmtype_1, split='train', instructions_only=True), cfg.n_train)
    data2_train = random.sample(load_dataset_split(harmtype=cfg.harmtype_2, split='train', instructions_only=True), cfg.n_train)
    data3_train = random.sample(load_dataset_split(harmtype=cfg.harmtype_3, split='train', instructions_only=True), cfg.n_train)
    data1_val = random.sample(load_dataset_split(harmtype=cfg.harmtype_1, split='val', instructions_only=True), cfg.n_val)
    data2_val = random.sample(load_dataset_split(harmtype=cfg.harmtype_2, split='val', instructions_only=True), cfg.n_val)
    data3_val = random.sample(load_dataset_split(harmtype=cfg.harmtype_3, split='val', instructions_only=True), cfg.n_val)
    return data1_train, data2_train, data3_train, data1_val, data2_val, data3_val

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val, or_train, or_val, system_harm, system_or, system_harmless):
    def filter_examples(dataset, scores, threshold, comparison_fn):
        if not dataset: return []
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison_fn(score, threshold)]

    if cfg.get('filter_train', False):
        print("Filtering training data based on refusal scores...")
        if harmful_train:
            harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, system=system_harm, batch_size=cfg.batch_size)
            harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        if harmless_train:
            harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, system=system_harmless, batch_size=cfg.batch_size)
            harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x <= y)
        print(f"Filtered harmful_train: {len(harmful_train)}, harmless_train: {len(harmless_train)}")

    if cfg.get('filter_or', False) and or_train:
        print("Filtering OR training data based on refusal scores...")
        or_train_scores = get_refusal_scores(model_base.model, or_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, system=system_or, batch_size=cfg.batch_size)
        or_train = filter_examples(or_train, or_train_scores, 0, lambda x, y: x > y)
        print(f"Filtered or_train: {len(or_train)}")

    if cfg.get('filter_val', False):
        print("Filtering validation data based on refusal scores...")
        if harmful_val:
            harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, system=system_harm, batch_size=cfg.batch_size)
            harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        if harmless_val:
            harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, system=system_harmless, batch_size=cfg.batch_size)
            harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x <= y)
        if or_val:
            or_val_scores = get_refusal_scores(model_base.model, or_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, system=system_or, batch_size=cfg.batch_size)
            or_val = filter_examples(or_val, or_val_scores, 0, lambda x, y: x > y)
        print(f"Filtered harmful_val: {len(harmful_val)}, harmless_val: {len(harmless_val)}, or_val: {len(or_val)}")
        
    return harmful_train, harmless_train, harmful_val, harmless_val, or_train, or_val

# --- Vector Generation & Selection (from original) ---
def generate_and_save_candidate_directions(cfg, model_base, data_positive, data_negative, system, direction_type_label=""):
    artifact_subdir = os.path.join(cfg.artifact_path, f'generate_directions_{direction_type_label}')
    os.makedirs(artifact_subdir, exist_ok=True)
    if not data_positive or not data_negative:
        print(f"Warning: Empty positive or negative dataset for generating directions: {direction_type_label}. Skipping.")
        return torch.empty(0)
    print(f"Generating directions for '{direction_type_label}' using {len(data_positive)} positive and {len(data_negative)} negative examples...")
    mean_diffs = generate_directions(system, model_base, data_positive, data_negative, artifact_dir=artifact_subdir, batch_size=cfg.batch_size)
    save_path = os.path.join(artifact_subdir, 'mean_diffs.pt')
    torch.save(mean_diffs, save_path)
    print(f"Candidate directions saved to {save_path}")
    return mean_diffs

def select_and_save_direction(cfg, model_base, val_positive, val_negative, candidate_directions, pair_name_label, mode, kl_threshold, top_n):
    select_dir_path = os.path.join(cfg.artifact_path, f"select_direction_{pair_name_label}_{mode}")
    os.makedirs(select_dir_path, exist_ok=True)
    print(f"Selecting direction for '{pair_name_label}' in '{mode}' mode...")
    if not val_positive or not val_negative:
        print(f"Warning: Empty validation set for selecting direction '{pair_name_label}'. Skipping selection.")
        return [None]*top_n, [None]*top_n, [torch.empty(0)]*top_n 
    if candidate_directions.numel() == 0:
        print(f"Warning: No candidate directions to select from for '{pair_name_label}'. Skipping selection.")
        return [None]*top_n, [None]*top_n, [torch.empty(0)]*top_n

    positions, layers, directions = select_direction(
        cfg, model_base, val_positive, val_negative, candidate_directions,
        pair_name_label, kl_threshold=kl_threshold, artifact_dir=select_dir_path,
        mode=mode, top_n=top_n, batch_size=cfg.batch_size,
    )
    metadata_to_save = []
    saved_directions = []
    if not isinstance(positions, list): 
        positions, layers, directions = [positions], [layers], [directions]
    for i in range(len(positions)):
        metadata_to_save.append({"pos": positions[i], "layer": layers[i]})
        torch.save(directions[i], f'{cfg.artifact_path}/direction_{pair_name_label}_{mode}_top{i+1}.pt')
        saved_directions.append(directions[i])
    with open(f'{cfg.artifact_path}/direction_metadata_{pair_name_label}_{mode}.json', "w") as f:
        json.dump(metadata_to_save, f, indent=4)
    print(f"Selected {len(positions)} direction(s) for '{pair_name_label}' in '{mode}' mode.")
    return positions, layers, saved_directions

# --- Orthogonalization (from original) ---
def ortho_refusal_directions(cfg, directions_to_orthogonalize, reference_directions_batch):
    epsilon = 1e-8
    if directions_to_orthogonalize.numel() == 0 or reference_directions_batch.numel() == 0:
        print("Warning: Empty directions for orthogonalization. Returning original directions_to_orthogonalize.")
        return directions_to_orthogonalize
    if directions_to_orthogonalize.shape != reference_directions_batch.shape:
         print(f"Warning: Shapes for orthogonalization differ. Dims1: {directions_to_orthogonalize.shape}, Dims2: {reference_directions_batch.shape}. Attempting broadcast or specific logic if intended for single ref vector.")
         # Add specific broadcasting logic if reference_directions_batch is a single selected vector
         if reference_directions_batch.ndim < directions_to_orthogonalize.ndim: # e.g. ref is [D] or [L,D] and dirs_to_orth is [P,L,D]
            try:
                reference_directions_batch = reference_directions_batch.expand_as(directions_to_orthogonalize)
            except RuntimeError:
                print("ERROR: Could not broadcast reference_directions_batch to match directions_to_orthogonalize. Orthogonalization may fail or be incorrect.")
                return directions_to_orthogonalize # Or raise error

    print(f"Orthogonalizing directions with lambda={cfg.ortho_lambda}...")
    dot_product = torch.sum(directions_to_orthogonalize * reference_directions_batch, dim=-1, keepdim=True)
    norm_sq_reference = torch.sum(reference_directions_batch * reference_directions_batch, dim=-1, keepdim=True)
    scaling_factor = dot_product / (norm_sq_reference + epsilon)
    projection = scaling_factor * reference_directions_batch
    orthogonalized_directions = directions_to_orthogonalize - cfg.ortho_lambda * projection
    norm_orthogonalized = torch.norm(orthogonalized_directions, p=2, dim=-1, keepdim=True)
    normalized_orthogonalized_directions = orthogonalized_directions / (norm_orthogonalized + epsilon)
    return normalized_orthogonalized_directions

# --- EBM Training Data Preparation Helpers ---
global_captured_activations_hook_data = {} # Global to store activations from hook
def _activation_capture_hook_fn_factory(layer_idx, position_indices_list):
    def _hook_fn(module, input_act_tuple):
        activation_tensor = input_act_tuple[0].clone().detach()
        batch_activations_at_target_pos_list = []
        for pos_idx_relative in position_indices_list:
            actual_pos_idx = activation_tensor.shape[1] + pos_idx_relative if pos_idx_relative < 0 else pos_idx_relative
            if 0 <= actual_pos_idx < activation_tensor.shape[1]:
                batch_activations_at_target_pos_list.append(activation_tensor[:, actual_pos_idx, :])
            else:
                batch_activations_at_target_pos_list.append(torch.zeros_like(activation_tensor[:, 0, :]))
        
        if len(batch_activations_at_target_pos_list) > 1:
            batch_activations_at_target = torch.mean(torch.stack(batch_activations_at_target_pos_list), dim=0)
        elif batch_activations_at_target_pos_list:
            batch_activations_at_target = batch_activations_at_target_pos_list[0]
        else:
            batch_activations_at_target = torch.zeros((activation_tensor.shape[0], activation_tensor.shape[2]), device=activation_tensor.device, dtype=activation_tensor.dtype)

        if layer_idx not in global_captured_activations_hook_data:
            global_captured_activations_hook_data[layer_idx] = []
        global_captured_activations_hook_data[layer_idx].append(batch_activations_at_target.cpu())
    return _hook_fn

def _extract_activations_for_ebm_training(
    llm_model_base, prompts_for_extraction: List[str], 
    target_layer: int, target_positions: List[int], 
    batch_size: int, device: str, system_prompt: Optional[str] = None
):
    llm_model_base.model.eval()
    all_extracted_activations = []
    hook_module = llm_model_base.model_block_modules[target_layer]
    hook_fn = _activation_capture_hook_fn_factory(target_layer, target_positions)
    
    print(f"Extracting activations from layer {target_layer} at positions {target_positions} for {len(prompts_for_extraction)} prompts...")
    for i in tqdm(range(0, len(prompts_for_extraction), batch_size), desc=f"Extracting acts (L{target_layer}, P{target_positions})"):
        batch_prompts = prompts_for_extraction[i:i+batch_size]
        inputs = llm_model_base.tokenize_instructions_fn(instructions=batch_prompts, system=system_prompt)
        
        global_captured_activations_hook_data.clear()
        handle = hook_module.register_forward_pre_hook(hook_fn)
        
        with torch.no_grad():
            llm_model_base.model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device)
            )
        handle.remove()
        
        if target_layer in global_captured_activations_hook_data and global_captured_activations_hook_data[target_layer]:
            batch_layer_activations = torch.cat(global_captured_activations_hook_data[target_layer], dim=0)
            all_extracted_activations.append(batch_layer_activations)
            
    if not all_extracted_activations: return torch.empty(0)
    return torch.cat(all_extracted_activations, dim=0)

def _classify_llm_response_for_ebm(response_text: str, is_prompt_harmful_context: bool) -> Optional[str]:
    is_refusal = substring_matching_judge_fn(response_text) # True if it IS a refusal
    if is_prompt_harmful_context:
        return 'positive_tr' if is_refusal else 'negative_tr' # positive_tr = True Refusal (good), negative_tr = Harmful Compliance (bad)
    else:
        return 'negative_fr' if is_refusal else 'positive_fr' # negative_fr = False Refusal (bad), positive_fr = Compliant Safe (good)

# --- EBM Training Function ---
def train_and_save_ebm_if_needed(cfg, llm_model_base: ModelBase, force_retrain: bool = False):
    if not (hasattr(cfg, 'ebm_params') and cfg.ebm_params):
        print("No ebm_params in config. Skipping EBM training.")
        return None, None

    # Helper function to reconstruct response data
    def _reconstruct_responses_with_metadata(
        original_data_list: List[Dict[str, Any]],
        generation_outputs: List[Any], # Expected: List[Dict] or List[str]
        source_prompts_list: Optional[List[str]] = None # Optional: if prompts used for gen differ from original_data_list's instructions
    ) -> List[Dict[str, Any]]:
        reconstructed_list = []
        if len(generation_outputs) != len(original_data_list):
            print(f"Error: Mismatch in lengths between generation inputs ({len(original_data_list)}) and outputs ({len(generation_outputs)}). Cannot reliably reconstruct.")
            return []

        for i, original_data_item in enumerate(original_data_list):
            generated_item = generation_outputs[i]
            
            response_text = ""
            # Use prompt from source_prompts_list if available, else original instruction
            prompt_text = source_prompts_list[i] if source_prompts_list and i < len(source_prompts_list) else original_data_item.get('instruction', "")

            if isinstance(generated_item, dict):
                response_text = generated_item.get('response', "").strip()
                # If generated_item contains 'prompt', it might be more specific (e.g. formatted)
                prompt_text = generated_item.get('prompt', prompt_text)
            elif isinstance(generated_item, str): # If generation_outputs is List[str]
                response_text = generated_item.strip()
            else:
                print(f"Warning: Unexpected output format from generation for item {i}: {generated_item}")

            reconstructed_list.append({
                'prompt': prompt_text,
                'response': response_text,
                'is_harmful_context': original_data_item['is_harmful_context'],
                'source_type': original_data_item['source_type']
            })
        return reconstructed_list

    ebm_model_fr, ebm_model_tr = None, None
    ebm_fr_path_template = cfg.ebm_params.get('ebm_fr_save_path')

    if not ebm_fr_path_template:
        print("Warning: ebm_params.ebm_fr_save_path not defined. Cannot train or load EBM_FR.")
        return None, None

    target_layer_fr = cfg.ebm_params.get('ebm_target_layer')
    target_positions_fr_str = str(cfg.ebm_params.get('ebm_target_positions', "-1")) # Ensure string
    target_positions_fr_filename_part = target_positions_fr_str.replace(",", "_").replace("-", "neg")
    
    try:
        resolved_ebm_fr_path = ebm_fr_path_template.format(
            ebm_target_layer=target_layer_fr,
            ebm_target_positions_filename_part=target_positions_fr_filename_part
        )
    except KeyError as e:
        raise ValueError(f"Error formatting ebm_fr_save_path ('{ebm_fr_path_template}'). Missing key for placeholder: {e}. Check config.")

    if not os.path.exists(resolved_ebm_fr_path) or force_retrain:
        print(f"EBM_FR model not found at {resolved_ebm_fr_path} or force_retrain=True. Starting training...")
        
        # 1. Collect (prompt, response, ebm_label) triples
        # MODIFIED: Load from ebm_fr_positive_datasets list
        positive_dataset_names = cfg.ebm_params.get('ebm_fr_positive_datasets', [])
        if not positive_dataset_names:
            # Fallback or error if old config key ebm_pseudo_harmful_data was also removed and new one not present
            old_pseudo_harmful_data_key = cfg.ebm_params.get('ebm_pseudo_harmful_data')
            if old_pseudo_harmful_data_key:
                print(f"Warning: ebm_fr_positive_datasets not found, falling back to ebm_pseudo_harmful_data: {old_pseudo_harmful_data_key}")
                positive_dataset_names = [old_pseudo_harmful_data_key]
            else:
                raise ValueError("EBM training requires ebm_fr_positive_datasets or ebm_pseudo_harmful_data to be defined in config.")

        pseudo_harmful_texts = []
        for ds_name in positive_dataset_names:
            # Assuming num_initial_prompts is per dataset for now, or adjust logic if it's total
            # load_dataset_split also needs to be robust to different dataset types or use load_dataset
            try:
                prompts = load_dataset_split(harmtype=ds_name, split='train', instructions_only=True)
                # Sample cfg.ebm_params.num_initial_prompts from each dataset, if larger
                if len(prompts) > cfg.ebm_params.num_initial_prompts:
                    prompts = random.sample(prompts, cfg.ebm_params.num_initial_prompts)
                pseudo_harmful_texts.extend(prompts)
            except Exception as e:
                print(f"Warning: Could not load/sample dataset '{ds_name}' for EBM training. Error: {e}")
        
        # Ensure pseudo_harmful_texts is not overly large if num_initial_prompts was meant to be a total cap
        # For simplicity, current logic takes up to num_initial_prompts from *each* listed dataset.
        # If a total cap is desired, sample from the aggregated pseudo_harmful_texts here.
        # Example: if len(pseudo_harmful_texts) > cfg.ebm_params.get('total_max_positive_prompts', some_default_large_number):
        # pseudo_harmful_texts = random.sample(pseudo_harmful_texts, cfg.ebm_params.get('total_max_positive_prompts'))

        if not pseudo_harmful_texts:
            raise ValueError("No pseudo-harmful prompts collected for EBM training. Check dataset names and availability.")

        harmless_texts = load_dataset_split(harmtype=cfg.ebm_params.ebm_harmless_data, split='train', instructions_only=True)
        if len(harmless_texts) > cfg.ebm_params.num_initial_prompts: # Sample harmless texts too
            harmless_texts = random.sample(harmless_texts, cfg.ebm_params.num_initial_prompts)

        data_for_classification = []
        for p_text in pseudo_harmful_texts: data_for_classification.append({'instruction': p_text, 'is_harmful_context': False, 'source_type': 'pseudo_harmful'})
        for p_text in harmless_texts: data_for_classification.append({'instruction': p_text, 'is_harmful_context': False, 'source_type': 'harmless'})
        
        print(f"Generating initial LLM responses for {len(data_for_classification)} prompts to create EBM training data...")
        
        # Initialize vLLM engine with config parameters
        from vllm import LLMEngine, EngineArgs, SamplingParams

        # Get the cache directory for model weights
        cache_dir = os.path.join(os.path.dirname(resolved_ebm_fr_path), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)

        engine_args = EngineArgs(
            model=cfg.model_path,
            download_dir=cache_dir,
            tensor_parallel_size=cfg.ebm_params.get('vllm_tensor_parallel_size', 1),
            max_num_batched_tokens=cfg.ebm_params.get('vllm_max_num_batched_tokens', 4096),
            max_num_seqs=cfg.ebm_params.get('vllm_max_num_seqs', 256),
            gpu_memory_utilization=cfg.ebm_params.get('vllm_gpu_memory_utilization', 0.90),
            block_size=cfg.ebm_params.get('vllm_block_size', 16),
            trust_remote_code=True,  # Often needed for newer models
            dtype='auto',  # Let vLLM automatically determine the dtype
            quantization=None,  # No quantization by default
            enforce_eager=True,  # More reliable for first run
            max_model_len=cfg.ebm_params.get('vllm_max_model_len', 4096), # Default to 4096, adjust in cfg.yaml per model (e.g., Llama2-7B is 4096)
        )

        llm_responses = [] # Ensure llm_responses is initialized
        engine_initialized_successfully = False
        try:
            print("Initializing vLLM engine...")
            engine = LLMEngine.from_engine_args(engine_args)
            engine_initialized_successfully = True
        except Exception as e:
            engine_init_error_msg = str(e)
            print(f"Error initializing vLLM engine: {engine_init_error_msg}")
            if "User-specified max_model_len" in engine_init_error_msg and "is greater than the derived max_model_len" in engine_init_error_msg:
                try:
                    derived_len_str = engine_init_error_msg.split("derived max_model_len (max_position_embeddings=")[1].split(" ")[0].split("=")[-1].split(")")[0]
                    print(f"CONTEXT: The model's derived max_model_len is likely {derived_len_str}.")
                    print(f"SUGGESTION: Try setting 'vllm_max_model_len: {derived_len_str}' in your config for this model ('{cfg.model_alias}'),")
                    print("OR set the environment variable VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 if you intend to override (use with caution).")
                except Exception: # Parsing failed, just show generic message
                    print("INFO: Check 'vllm_max_model_len' in your config against the model's capabilities or consider VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.")
            
            print("Falling back to standard Hugging Face generation method...")
            llm_gen_batch_size = cfg.ebm_params.get('llm_gen_batch_size_for_ebm_data', cfg.batch_size) 
            
            standard_hf_outputs = llm_model_base.generate_completions(
                data_for_classification, 
                batch_size=llm_gen_batch_size, 
                max_new_tokens=cfg.ebm_params.max_new_tokens_for_ebm_data, 
                system=cfg.get('system')
            )
            llm_responses = _reconstruct_responses_with_metadata(data_for_classification, standard_hf_outputs)

        if engine_initialized_successfully:
            # Check if llm_model_base.refusal_toks contains integers (token IDs)
            # If they are strings, use `stop=llm_model_base.refusal_toks`
            # If they are integers (token_ids), use `stop_token_ids=llm_model_base.refusal_toks`
            stop_param_args = {}
            if llm_model_base.refusal_toks and all(isinstance(tok, int) for tok in llm_model_base.refusal_toks):
                stop_param_args['stop_token_ids'] = llm_model_base.refusal_toks
            elif llm_model_base.refusal_toks: # Assuming it's a list of strings if not all ints
                stop_param_args['stop'] = llm_model_base.refusal_toks
            # If refusal_toks is None or empty, no stop parameter is added, which is fine.

            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic
                max_tokens=cfg.ebm_params.max_new_tokens_for_ebm_data,
                **stop_param_args
            )
            
            # Prepare prompts with system prompt if needed (vLLM path)
            vllm_prompts_list = []
            system_prompt_for_vllm = cfg.get('system')
            for item_for_vllm in data_for_classification:
                if system_prompt_for_vllm:
                    current_vllm_prompt = f"{system_prompt_for_vllm}\\n\\nHuman: {item_for_vllm['instruction']}\\n\\nAssistant:"
                else:
                    current_vllm_prompt = f"Human: {item_for_vllm['instruction']}\\n\\nAssistant:"
                vllm_prompts_list.append(current_vllm_prompt)
            
            print(f"Generating responses using vLLM for {len(vllm_prompts_list)} prompts...")
            try:
                vllm_engine_outputs = engine.generate(vllm_prompts_list, sampling_params)
                
                # Process vLLM outputs and create labeled data
                # The _reconstruct_responses_with_metadata helper expects List[Dict] or List[str]
                # vllm_engine_outputs is List[RequestOutput]. We need to extract response texts.
                extracted_vllm_response_texts = [out.outputs[0].text for out in vllm_engine_outputs]
                llm_responses = _reconstruct_responses_with_metadata(data_for_classification, extracted_vllm_response_texts, source_prompts_list=vllm_prompts_list)

            except Exception as e_vllm_gen:
                print(f"Error during vLLM generation: {str(e_vllm_gen)}")
                print("Falling back to standard Hugging Face generation method...")
                llm_gen_batch_size_fallback = cfg.ebm_params.get('llm_gen_batch_size_for_ebm_data', cfg.batch_size)
                
                standard_hf_outputs_fallback = llm_model_base.generate_completions(
                    data_for_classification, 
                    batch_size=llm_gen_batch_size_fallback, 
                    max_new_tokens=cfg.ebm_params.max_new_tokens_for_ebm_data, 
                    system=cfg.get('system')
                )
                llm_responses = _reconstruct_responses_with_metadata(data_for_classification, standard_hf_outputs_fallback)
            finally:
                # Clean up vLLM engine if it was created in this scope
                if 'engine' in locals() and engine_initialized_successfully: 
                    del engine
                    # if 'vllm_engine_outputs' in locals(): del vllm_engine_outputs # Not strictly needed as it's local to try
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # llm_responses is now populated either by vLLM or fallback, with consistent structure.
        if not llm_responses:
             raise ValueError("EBM training data generation failed (llm_responses is empty). Check logs for errors.")

        labeled_prompts_for_activation_extraction = []
        for item in llm_responses:
            ebm_label = _classify_llm_response_for_ebm(item['response'], item['is_harmful_context'])
            if ebm_label in ['positive_fr', 'negative_fr']:
                labeled_prompts_for_activation_extraction.append((item['prompt'], ebm_label))
        
        if not labeled_prompts_for_activation_extraction:
            raise ValueError("No suitable prompts found after classification for EBM_FR activation extraction.")

        prompts_to_extract = [item[0] for item in labeled_prompts_for_activation_extraction]
        target_positions_list_fr = [int(p.strip()) for p in target_positions_fr_str.split(',')]
        
        extracted_acts_tensor = _extract_activations_for_ebm_training(
            llm_model_base, prompts_to_extract, target_layer_fr, 
            target_positions_list_fr, cfg.ebm_params.activation_extraction_batch_size, cfg.device,
            system_prompt=cfg.get('system', None) # Pass system prompt if used for consistency
        )

        if extracted_acts_tensor.numel() == 0: raise ValueError("Failed to extract any activations for EBM_FR training.")

        X_fr_pos_list, X_fr_neg_list = [], []
        for idx, (_, ebm_label) in enumerate(labeled_prompts_for_activation_extraction):
            if idx < extracted_acts_tensor.shape[0]:
                if ebm_label == 'positive_fr': X_fr_pos_list.append(extracted_acts_tensor[idx])
                elif ebm_label == 'negative_fr': X_fr_neg_list.append(extracted_acts_tensor[idx])
        
        if not X_fr_pos_list or not X_fr_neg_list:
            raise ValueError(f"Insufficient positive ({len(X_fr_pos_list)}) or negative ({len(X_fr_neg_list)}) activation samples for EBM_FR training.")

        X_fr_pos = torch.stack(X_fr_pos_list).float()
        X_fr_neg = torch.stack(X_fr_neg_list).float()
        print(f"EBM_FR Training Data: Positive samples: {X_fr_pos.shape[0]}, Negative samples: {X_fr_neg.shape[0]}")

        min_len = min(X_fr_pos.shape[0], X_fr_neg.shape[0])
        if min_len == 0: raise ValueError("Not enough paired samples for EBM training.")
        ebm_dataset = TensorDataset(X_fr_pos[:min_len].to(cfg.device), X_fr_neg[:min_len].to(cfg.device))
        ebm_dataloader = DataLoader(ebm_dataset, batch_size=cfg.ebm_params.ebm_batch_size, shuffle=True)
        
        current_ebm_fr = SimpleEBM(input_dim=llm_model_base.model.config.hidden_size, hidden_dim=cfg.ebm_params.ebm_hidden_dim).to(cfg.device)
        optimizer = optim.Adam(current_ebm_fr.parameters(), lr=cfg.ebm_params.ebm_lr)
        
        current_ebm_fr.train()
        for epoch in range(cfg.ebm_params.ebm_epochs):
            total_loss = 0
            for x_p_batch, x_n_batch in tqdm(ebm_dataloader, desc=f"EBM_FR Train Epoch {epoch+1}"):
                optimizer.zero_grad()
                energy_p = current_ebm_fr(x_p_batch)
                energy_n = current_ebm_fr(x_n_batch)
                loss = torch.relu(cfg.ebm_params.ebm_margin + energy_p.mean() - energy_n.mean())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"EBM_FR Epoch {epoch+1} Avg Loss: {total_loss / len(ebm_dataloader):.4f}")
        
        os.makedirs(os.path.dirname(resolved_ebm_fr_path), exist_ok=True)
        torch.save(current_ebm_fr.state_dict(), resolved_ebm_fr_path)
        print(f"Trained EBM_FR model saved to {resolved_ebm_fr_path}")
        ebm_model_fr = current_ebm_fr
    else:
        print(f"Loading existing EBM_FR model from {resolved_ebm_fr_path}")
        ebm_model_fr = load_ebm_model(
            resolved_ebm_fr_path,
            input_dim=llm_model_base.model.config.hidden_size,
            hidden_dim=cfg.ebm_params.ebm_hidden_dim,
            device=cfg.device
        )
        if ebm_model_fr is None:
            raise ValueError(f"Failed to load EBM_FR from existing path: {resolved_ebm_fr_path}")
    
    # Clean up vLLM engine if it was created (redundant if already handled in the if block, but harmless)
    # if 'engine' in locals() and engine_initialized_successfully:
    #     del engine
    #     gc.collect()
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    return ebm_model_fr, ebm_model_tr


# --- Main Pipeline Function (Continued) ---
def run_pipeline(config_path, model_path_override=None, batch_size_override=None, force_retrain_ebm_cli=False):
    cfg = mmengine.Config.fromfile(config_path)
    
    if model_path_override is not None: cfg.model_path = model_path_override
    if batch_size_override is not None: cfg.batch_size = batch_size_override
    cfg.model_alias = os.path.basename(cfg.model_path)
    
    current_mode = cfg.get('mode', 'baseline' if cfg.get('baseline', False) else 'unknown_intervention') # Default to baseline if baseline=True
    
    # Correctly get force_retrain_ebm from cfg.ebm_params or CLI
    force_retrain_ebm = force_retrain_ebm_cli 
    if hasattr(cfg, 'ebm_params') and cfg.ebm_params is not None:
        force_retrain_ebm = force_retrain_ebm or cfg.ebm_params.get('force_retrain_ebm', False)


    if 'artifact_path' not in cfg or cfg.artifact_path is None:
        mode_suffix = current_mode
        if 'ebm' in mode_suffix and hasattr(cfg, 'ebm_params') and cfg.ebm_params:
            eta_val = cfg.ebm_params.get('ebm_eta', 'defEta')
            steps_val = cfg.ebm_params.get('ebm_num_gradient_steps', 'defSteps')
            layer_val = cfg.ebm_params.get('ebm_target_layer', 'defLyr')
            pos_val_str = str(cfg.ebm_params.get('ebm_target_positions', "-1")).replace(",", "_").replace("-", "neg")
            mode_suffix += f"_L{layer_val}_P{pos_val_str}_eta{eta_val}_steps{steps_val}"
        cfg.artifact_path = os.path.join("output", cfg.model_alias, mode_suffix)
    
    os.makedirs(cfg.artifact_path, exist_ok=True)
    sys.stdout = Logger(os.path.join(cfg.artifact_path, "output.log"))
    sys.stderr = Logger(os.path.join(cfg.artifact_path, "error.log"))
    
    print(f"--- Starting Pipeline for Mode: {current_mode} ---")
    print(f"Artifacts will be saved to: {cfg.artifact_path}")
    cfg.dump(os.path.join(cfg.artifact_path, 'config_run.yaml'))
    
    print(f"Loading LLM: {cfg.model_path}")
    model_base = construct_model_base(cfg.model_path)
    # model_base.model.to(cfg.get('device', 'cuda'))

    ebm_model_fr, ebm_model_tr = None, None
    if 'ebm' in current_mode:
        # ... (EBM loading logic, ensure EBMs are also moved to an appropriate device if not handled by load_ebm_model) ...
        # Example: if ebm_model_fr: ebm_model_fr.to(cfg.get('device', 'cuda')) 
        # Better: ebm_model_fr.to(next(model_base.model.parameters()).device) if model_base.model is on GPU
        # For EBMs, if they are small, placing them on the primary GPU (e.g., cuda:0) is usually fine.
        # The `load_ebm_model` function was sketched to include a device parameter.
        print("EBM mode detected. Attempting to train or load EBMs...")
        ebm_model_fr, ebm_model_tr = train_and_save_ebm_if_needed(cfg, model_base, force_retrain_ebm)
        if ebm_model_fr is None and 'ebm_fr_push' in current_mode: 
             raise ValueError("EBM_FR is required for this EBM mode but could not be loaded or trained.")
        if ebm_model_fr:
            ebm_model_fr.to(cfg.get('device', 'cuda')) # Move EBM to the main device
        if ebm_model_tr:
            ebm_model_tr.to(cfg.get('device', 'cuda')) # Move EBM to the main device


    print("Loading and sampling datasets for pipeline run...")
    or_train, harmless_train, harmful_train, \
    or_val, harmless_val, harmful_val = load_and_sample_datasets(cfg)
    
    print("Filtering datasets for pipeline run...")
    harmful_train, harmless_train, harmful_val, harmless_val, or_train, or_val = filter_data(
        cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val, 
        or_train, or_val, 
        system_harm=cfg.get('system_harm', cfg.get('system')), 
        system_or=cfg.get('system_or', cfg.get('system')), 
        system_harmless=cfg.get('system_harmless', cfg.get('system'))
    )

    or_bench_hard_data_prompts = random.sample(load_dataset_split(harmtype='or_bench_hard', split='test', instructions_only=True), cfg.n_test)
    harmful_data_test_prompts = random.sample(load_dataset_split(harmtype='harmful', split='test', instructions_only=True), cfg.n_test)
    
    # Convert to list of dicts for generate_completions
    or_bench_hard_data = [{'instruction': p, 'category': 'or_bench_hard'} for p in or_bench_hard_data_prompts]
    harmful_data_test = [{'instruction': p, 'category': 'harmful'} for p in harmful_data_test_prompts]


    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    intervention_fwd_pre_hooks, intervention_fwd_hooks = [], []

    if not cfg.get('baseline', False):
        if 'ebm' in current_mode:
            if ebm_model_fr is None: 
                raise ValueError("EBM_FR model is required for EBM modes but not available after train/load step.")
            
            print(f"Setting up EBM intervention hooks for mode: {current_mode}")
            ebm_intervention_layers = cfg.ebm_params.get('ebm_intervention_layers', [cfg.ebm_params.ebm_target_layer])
            ebm_intervention_positions_str = str(cfg.ebm_params.get('ebm_intervention_positions', str(cfg.ebm_params.ebm_target_positions)))
            ebm_intervention_positions = [int(p.strip()) for p in ebm_intervention_positions_str.split(',')]

            for layer_idx in ebm_intervention_layers:
                if not (0 <= layer_idx < model_base.model.config.num_hidden_layers):
                    print(f"Warning: EBM intervention layer {layer_idx} is out of bounds. Skipping.")
                    continue
                
                target_module = model_base.model_block_modules[layer_idx]
                ebm_hook = get_ebm_intervention_hook(
                    ebm_model_fr=ebm_model_fr,
                    target_layer_idx=layer_idx, current_hook_layer_idx=layer_idx,
                    position_indices=ebm_intervention_positions,
                    eta=cfg.ebm_params.ebm_eta,
                    num_gradient_steps=cfg.ebm_params.ebm_num_gradient_steps,
                    ebm_model_tr=ebm_model_tr, 
                    lambda_ebm_ortho=cfg.ebm_params.ebm_lambda_ortho,
                    device=model_base.model.device
                )
                intervention_fwd_pre_hooks.append((target_module, ebm_hook))
            print(f"Added {len(intervention_fwd_pre_hooks)} EBM intervention pre-hooks.")

            if 'harm_actadd' in current_mode: 
                print("Setting up True Refusal (harmful contrast) vector addition for hybrid EBM mode...")
                if not harmful_train or not harmless_train or not harmful_val or not harmless_val:
                    print("Warning: Missing data for harm_actadd component. Skipping harm_actadd.")
                else:
                    cand_harm_contrast_hybrid = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train, system=None, direction_type_label="harm_contrast_for_hybrid_ebm")
                    if cand_harm_contrast_hybrid.numel() > 0:
                        cand_harm_contrast_hybrid = cand_harm_contrast_hybrid[:, cfg.start_layer:,]
                        pos_h_hybrid, lyr_h_hybrid, dir_h_hybrid_list_of_lists = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, cand_harm_contrast_hybrid, "harm_add_hybrid", 'addition', cfg.steer_kl_threshold, 1)
                        
                        added_hooks_count = 0
                        for p_list, l_list, d_tensor_outer_list in zip(pos_h_hybrid, lyr_h_hybrid, dir_h_hybrid_list_of_lists):
                             # Assuming select_and_save_direction returns lists of positions, layers, and a list of direction tensors (even for top_n=1)
                             # If top_n=1, p_list, l_list will have 1 element, d_tensor_outer_list will have 1 tensor.
                             p_val = p_list[0] if isinstance(p_list, list) else p_list # Handle if not list for top_n=1
                             l_val = l_list[0] if isinstance(l_list, list) else l_list
                             d_tensor = d_tensor_outer_list[0] if isinstance(d_tensor_outer_list, list) and d_tensor_outer_list else d_tensor_outer_list

                             if d_tensor is not None and d_tensor.numel() > 0:
                                intervention_fwd_pre_hooks.append(
                                    (model_base.model_block_modules[l_val], get_activation_addition_input_pre_hook(vector=d_tensor, coeff=+cfg.addact_coeff))
                                )
                                added_hooks_count +=1
                        print(f"Added {added_hooks_count} harm_actadd pre-hooks for hybrid mode.")
                    else:
                        print("Warning: No candidate harm_contrast directions generated for hybrid EBM mode.")
        
        elif current_mode == 'or_ablation_harm_actadd' or current_mode == 'harm_ablation':
            # (This is the existing vector-based intervention logic from your original script)
            # ... (ensure this part is complete and correct as per your working version) ...
            print(f"Setting up original vector-based intervention: {current_mode}")
            # --- This is the original paper's logic for vector interventions ---
            if not all([harmful_train, harmless_train, harmful_val, harmless_val]):
                 raise ValueError("Missing datasets required for vector-based intervention.")
            
            candidate_directions_harm_contrast = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train, system=None, direction_type_label="harm_contrast_vec")
            if candidate_directions_harm_contrast.numel() == 0: raise ValueError("Failed to generate harm_contrast directions.")
            candidate_directions_harm_contrast = candidate_directions_harm_contrast[:, cfg.start_layer:,]
            positions_harm, layers_harm, directions_harm_outer = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions_harm_contrast, ['harmful', 'harmless'], 'addition', cfg.steer_kl_threshold, 1)

            temp_harm_actadd_fwd_pre_hooks = []
            actual_directions_harm = [] # Store the selected harm direction(s) for OR ortho
            for p_list, l_list, d_list in zip(positions_harm, layers_harm, directions_harm_outer):
                p_h, l_h, d_h = (p_list[0] if isinstance(p_list,list) else p_list, 
                                 l_list[0] if isinstance(l_list,list) else l_list, 
                                 d_list[0] if isinstance(d_list,list) else d_list) # Assuming top_n=1 for this path
                temp_harm_actadd_fwd_pre_hooks.append(
                    (model_base.model_block_modules[l_h], get_activation_addition_input_pre_hook(vector=d_h, coeff=+cfg.addact_coeff))
                )
                actual_directions_harm.append(d_h)


            if current_mode == 'harm_ablation':
                pre_h_abl, fwd_h_abl = [], []
                for d_h in actual_directions_harm: # Ablate the selected harm direction(s)
                    _pre, _fwd = get_all_direction_ablation_hooks(model_base, d_h, cfg.start_layer, cfg.ablation_coeff)
                    pre_h_abl.extend(_pre)
                    fwd_h_abl.extend(_fwd)
                intervention_fwd_pre_hooks.extend(pre_h_abl)
                intervention_fwd_hooks.extend(fwd_h_abl)
            
            if 'or_ablation' in current_mode: # This applies to 'or_ablation_harm_actadd'
                if not or_train or not or_val: raise ValueError("Missing OR datasets for or_ablation mode.")
                candidate_directions_or_contrast = generate_and_save_candidate_directions(cfg, model_base, or_train, harmless_train, system=cfg.system, direction_type_label="or_contrast_vec")
                if candidate_directions_or_contrast.numel() == 0: raise ValueError("Failed to generate or_contrast directions.")
                candidate_directions_or_contrast = candidate_directions_or_contrast[:, cfg.start_layer:, ]
                
                # Orthogonalize OR candidates against the *full set* of harm candidates
                cand_or_contrast_orth = ortho_refusal_directions(cfg, candidate_directions_or_contrast, candidate_directions_harm_contrast)
                
                positions_or, layers_or, directions_or_outer = select_and_save_direction(cfg, model_base, or_val, harmless_val, cand_or_contrast_orth, ['over_refusal', 'harmless'], 'ablation', cfg.ablate_kl_threshold, cfg.top_n)
                
                temp_or_ablation_fwd_pre_hooks, temp_or_ablation_fwd_hooks = [], []
                for p_list_o, l_list_o, d_list_o in zip(positions_or, layers_or, directions_or_outer):
                    # Iterate through directions if top_n > 1
                     for d_o_idx in range(len(d_list_o)):
                        p_o = p_list_o[d_o_idx] if isinstance(p_list_o, list) and len(p_list_o) > d_o_idx else p_list_o
                        l_o = l_list_o[d_o_idx] if isinstance(l_list_o, list) and len(l_list_o) > d_o_idx else l_list_o
                        d_o = d_list_o[d_o_idx]

                        pre_o, fwd_o = get_all_direction_ablation_hooks(model_base, d_o, cfg.start_layer, cfg.ablation_coeff) # Use ablation_coeff for OR
                        temp_or_ablation_fwd_pre_hooks.extend(pre_o)
                        temp_or_ablation_fwd_hooks.extend(fwd_o)
                
                intervention_fwd_pre_hooks.extend(temp_or_ablation_fwd_pre_hooks)
                intervention_fwd_hooks.extend(temp_or_ablation_fwd_hooks)
            
            if current_mode == 'or_ablation_harm_actadd':
                 intervention_fwd_pre_hooks.extend(temp_harm_actadd_fwd_pre_hooks)

        else:
            print(f"Warning: Mode '{current_mode}' is not 'baseline' and not a recognized intervention mode. Running with empty intervention hooks.")
    
    active_pre_hooks = intervention_fwd_pre_hooks if not cfg.get('baseline', False) else baseline_fwd_pre_hooks
    active_fwd_hooks = intervention_fwd_hooks if not cfg.get('baseline', False) else baseline_fwd_hooks
    eval_identifier = current_mode

    print(f"\n--- Generating and Evaluating Completions for Mode: {eval_identifier} ---")
    
    datasets_to_eval_completions = []
    if hasattr(cfg, 'jailbreak_evaluation_datasets'): datasets_to_eval_completions.extend(cfg.jailbreak_evaluation_datasets)
    if hasattr(cfg, 'over_refusal_evaluation_datasets'): datasets_to_eval_completions.extend(cfg.over_refusal_evaluation_datasets)
    datasets_to_eval_completions = sorted(list(set(datasets_to_eval_completions)))

    test_data_cache = {
        'harmful': harmful_data_test,
        'or_bench_hard': or_bench_hard_data,
    }

    for ds_name in datasets_to_eval_completions:
        current_test_data_list_of_strings_or_dicts = test_data_cache.get(ds_name)
        current_test_data_for_gen = None

        if current_test_data_list_of_strings_or_dicts:
            if isinstance(current_test_data_list_of_strings_or_dicts[0], str): # List of strings
                 current_test_data_for_gen = [{'instruction': item, 'category': ds_name} for item in current_test_data_list_of_strings_or_dicts]
            elif isinstance(current_test_data_list_of_strings_or_dicts[0], dict): # Already list of dicts
                 current_test_data_for_gen = current_test_data_list_of_strings_or_dicts
                 # Ensure 'category' is present if not already
                 for item in current_test_data_for_gen: item.setdefault('category', ds_name)
            else:
                print(f"Warning: Test data for {ds_name} is in an unrecognized format. Will attempt dynamic load.")
        
        generate_and_save_completions_for_dataset(
            cfg, model_base, active_pre_hooks, active_fwd_hooks, 
            eval_identifier, ds_name, dataset=current_test_data_for_gen, system=cfg.get('system', None)
        )
        
        current_eval_methodologies = []
        if ds_name in cfg.get('jailbreak_evaluation_datasets', []):
            current_eval_methodologies.extend(cfg.get('jailbreak_eval_methodologies', ['substring_matching']))
        if ds_name in cfg.get('over_refusal_evaluation_datasets', []): 
            current_eval_methodologies.extend(cfg.get('refusal_eval_methodologies', ['substring_matching']))
        current_eval_methodologies = list(set(current_eval_methodologies))

        if current_eval_methodologies:
            evaluate_completions_and_save_results_for_dataset(cfg, eval_identifier, ds_name, eval_methodologies=current_eval_methodologies)
        else:
            print(f"No evaluation methodologies specified for dataset: {ds_name}")

    print(f"\n--- Running LM Eval Harness for mode: {eval_identifier} ---")
    with add_hooks(module_forward_pre_hooks=active_pre_hooks, module_forward_hooks=active_fwd_hooks):
        eval_harness_results = eval_harness(cfg, model_base, eval_identifier)
        if eval_harness_results: 
            print(f"LM Eval Harness Results for {eval_identifier} (summary):")
            # Attempt to print main results if available
            for task_type, results_obj in eval_harness_results.items():
                if isinstance(results_obj, dict) and 'results' in results_obj:
                    print(f"  {task_type.upper()} Tasks:")
                    for task_name, metrics in results_obj['results'].items():
                        print(f"    {task_name}: {metrics}")
                elif isinstance(results_obj, dict): # Fallback if structure is slightly different
                    print(f"  {task_type.upper()} (raw): {results_obj}")


    print(f"--- Pipeline for mode {eval_identifier} completed. Results in {cfg.artifact_path} ---")
    
    del model_base.model
    del model_base
    if ebm_model_fr: del ebm_model_fr
    if ebm_model_tr: del ebm_model_tr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(
        config_path=args.config_path, 
        model_path_override=args.model_path, 
        batch_size_override=args.batch_size,
        force_retrain_ebm_cli=args.force_retrain_ebm
    )
