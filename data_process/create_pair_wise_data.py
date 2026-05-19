import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import pickle
from pathlib import Path
from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import load_from_disk
from huggingface_hub import login
from eval.criteria import REWARDBENCH_CONTEXT_MAP
# login()


def _extract_ultra_score(candidate: dict, attribute: str):
    """Best-effort extraction of UltraFeedback attribute score from completion-level metadata."""
    base_attr = attribute.replace("ultrafeedback-", "")
    base_attr_us = base_attr.replace("-", "_")

    def _to_float(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return float(value)
            except ValueError:
                return None
        if isinstance(value, dict):
            # Common UltraFeedback schema: annotations[dimension]["Rating"]
            for key in ("Rating", "rating", "score", "Score", "value"):
                if key in value:
                    parsed = _to_float(value[key])
                    if parsed is not None:
                        return parsed
        return None

    score_containers = [
        candidate.get("annotations"),
        candidate.get("scores"),
        candidate.get("score"),
        candidate.get("fine-grained_score"),
    ]
    lookup_keys = [
        base_attr,
        base_attr_us,
        attribute,
        attribute.replace("-", "_"),
    ]
    for container in score_containers:
        if isinstance(container, dict):
            for key in lookup_keys:
                if key in container:
                    parsed = _to_float(container[key])
                    if parsed is not None:
                        return parsed
    # Some variants store raw top-level fields on each candidate row
    for key in lookup_keys:
        if key in candidate:
            parsed = _to_float(candidate[key])
            if parsed is not None:
                return parsed
    return None


def _load_or_build_ultrafeedback_original_content(dataset_path: str):
    cache_path = Path("./dataset/ultrafeedback_original_content.pkl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with open(cache_path, "rb") as file:
            return pickle.load(file)

    print(f"[Info] {cache_path} not found, building from {dataset_path} ...")
    # Use a writable local cache to avoid permission issues with system-level HF cache locks.
    local_cache = "./.hf_datasets_cache"
    try:
        ds = load_dataset(dataset_path, split="train", cache_dir=local_cache)
    except Exception as e:
        print(f"[Warn] load_dataset failed ({type(e).__name__}: {e}). Trying local cached arrow shards...")
        local_root = Path(local_cache)
        home_root = Path.home() / ".cache" / "huggingface" / "datasets"
        shard_paths = sorted(local_root.glob("openbmb___ultra_feedback/default/0.0.0/*/ultra_feedback-train-*.arrow"))
        if not shard_paths:
            shard_paths = sorted(home_root.glob("openbmb___ultra_feedback/default/0.0.0/*/ultra_feedback-train-*.arrow"))
        if not shard_paths:
            raise FileNotFoundError(
                "Could not load UltraFeedback from hub and no local arrow shards found in "
                f"{local_root} or {home_root}."
            ) from e
        shards = [Dataset.from_file(str(path)) for path in shard_paths]
        ds = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
    attributes = [
        "ultrafeedback-helpfulness",
        "ultrafeedback-honesty",
        "ultrafeedback-instruction-following",
        "ultrafeedback-truthfulness",
    ]
    rows = []
    for item in tqdm(ds, desc="Building ultrafeedback_original_content"):
        prompt = item.get("prompt") or item.get("instruction")
        source = item.get("source", "unknown")
        completions = item.get("completions")
        if not isinstance(completions, list):
            continue
        for cand in completions:
            if not isinstance(cand, dict):
                continue
            content = cand.get("content") or cand.get("response") or cand.get("text")
            if not isinstance(content, str):
                continue
            row = {"prompt": prompt, "source": source, "content": content}
            valid = True
            for attr in attributes:
                value = _extract_ultra_score(cand, attr)
                if value is None:
                    valid = False
                    break
                row[attr] = value
            if valid:
                rows.append(row)

    if not rows:
        raise ValueError(
            "Failed to build ultrafeedback rows from dataset. "
            "Schema might differ from expected UltraFeedback format."
        )

    with open(cache_path, "wb") as file:
        pickle.dump(rows, file)
    print(f"[Info] Wrote {len(rows)} rows to {cache_path}")
    return rows

def create_pairwise_dataset_per_attribute_helpsteer2(dataset_path):
    ds1 = load_dataset(dataset_path)['train'].shuffle(seed=0)
    ds2 = load_dataset(dataset_path)['validation'].shuffle(seed=0)

    df = pd.DataFrame(ds1).reset_index().rename(columns={'index': 'original_index'})
    
    attributes = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
    pairwise_data = []
    cnt_pair_wise = {key:0 for key in attributes}
    cnt_pair_wise_test = {key:0 for key in attributes}
    training_set = []
    eval_set = []
    test_set = []

    for attr in attributes:
        assert attr in df.columns, f"Missing attribute in dataset: {attr}"
        cnt = 0

        for prompt, group in tqdm(df.groupby('prompt')):
            sorted_group = group.sort_values(by=attr, ascending=False)

            for i in range(len(sorted_group) - 1):
                chosen = sorted_group.iloc[i]
                for j in range(i + 1, len(sorted_group)):
                    rejected = sorted_group.iloc[j]
                    if chosen[attr] >= rejected[attr] + 1 and cnt < 500:
                        test_set.append({
                            'prompt': prompt,
                            'chosen': chosen['response'],
                            'rejected': rejected['response'],
                            'chosen_index': chosen['original_index'],
                            'rejected_index': rejected['original_index'],
                            'attribute': attr,  # Include the attribute in the pair data
                            'chosen_rating':chosen[attr],
                            'rejected_rating':rejected[attr],
                        })
                        cnt += 1
                        cnt_pair_wise_test[attr]+=1
                    elif chosen[attr] >= rejected[attr] + 1:  # Ensure a meaningful difference in score
                        training_set.append({
                            'prompt': prompt,
                            'chosen': chosen['response'],
                            'rejected': rejected['response'],
                            'chosen_index': chosen['original_index'],
                            'rejected_index': rejected['original_index'],
                            'attribute': attr,  # Include the attribute in the pair data
                            'chosen_rating':chosen[attr],
                            'rejected_rating':rejected[attr],
                        })
                        cnt_pair_wise[attr]+=1
                        cnt += 1

    df = pd.DataFrame(ds2).reset_index().rename(columns={'index': 'original_index'})
    for attr in attributes:
        assert attr in df.columns, f"Missing attribute in dataset: {attr}"
        cnt = 0

        for prompt, group in tqdm(df.groupby('prompt')):
            sorted_group = group.sort_values(by=attr, ascending=False)

            for i in range(len(sorted_group) - 1):
                chosen = sorted_group.iloc[i]
                for j in range(i + 1, len(sorted_group)):
                    rejected = sorted_group.iloc[j]
                    if chosen[attr] >= rejected[attr] + 1:
                        test_set.append({
                            'prompt': prompt,
                            'chosen': chosen['response'],
                            'rejected': rejected['response'],
                            'chosen_index': chosen['original_index'],
                            'rejected_index': rejected['original_index'],
                            'attribute': attr,  # Include the attribute in the pair data
                            'chosen_rating':chosen[attr],
                            'rejected_rating':rejected[attr],
                        })
                        cnt_pair_wise_test[attr]+=1
    
    print("Train:", cnt_pair_wise)
    print("Test:",cnt_pair_wise_test)
    pairwise_df_train = pd.DataFrame(training_set)
    pairwise_df_test = pd.DataFrame(test_set)

    train_dataset = Dataset.from_list(training_set)
    test_dataset = Dataset.from_list(test_set)

    dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

    dataset.save_to_disk("./dataset/helpsteer2_per_attribute_pairwise")
    # dataset.push_to_hub("pair/helpsteer2_per_attribute_pairwise")

    return pairwise_df_train, pairwise_df_test

def create_pairwise_dataset_per_attribute_ultra(dataset_path):
    ds1 = _load_or_build_ultrafeedback_original_content(dataset_path)

    df = pd.DataFrame(ds1).reset_index().rename(columns={'index': 'original_index'})
    
    attributes = ['ultrafeedback-helpfulness', 'ultrafeedback-honesty', 'ultrafeedback-instruction-following', 'ultrafeedback-truthfulness']
    pairwise_data = []
    training_set = []
    test_set = []

    sampled_prompts = []
    for source, group in df.groupby('source'):
        sampled_prompts.extend(
            group['prompt'].drop_duplicates().sample(frac=0.8, random_state=0).tolist()
        ) 
    train_df = df[df['prompt'].isin(sampled_prompts)]
    remaining_df = df[~df['prompt'].isin(sampled_prompts)]
    print('Num of Training Data',len(train_df))

    cnt_pair_wise = {key:0 for key in attributes}
    for attr in attributes:
        assert attr in df.columns, f"Missing attribute in dataset: {attr}"
        for prompt, group in tqdm(train_df.groupby('prompt')):
                sorted_group = group.sort_values(by=attr, ascending=False)
                for i in range(len(sorted_group) - 1):
                    chosen = sorted_group.iloc[i]
                    for j in range(i + 1, len(sorted_group)):
                        rejected = sorted_group.iloc[j]
                        if chosen[attr] >= rejected[attr] + 1:
                            training_set.append({
                                'prompt': prompt,
                                'chosen': chosen['content'],
                                'rejected': rejected['content'],
                                'attribute': attr,
                                'chosen_rating':chosen[attr],
                            'rejected_rating':rejected[attr],
                            })
                            cnt_pair_wise[attr] += 1
    print("Train:", cnt_pair_wise)

    cnt_pair_wise = {key:0 for key in attributes}
    for attr in attributes:
        assert attr in df.columns, f"Missing attribute in dataset: {attr}"
        for prompt, group in tqdm(remaining_df.groupby('prompt')):
            sorted_group = group.sort_values(by=attr, ascending=False)
            for i in range(len(sorted_group) - 1):
                chosen = sorted_group.iloc[i]
                for j in range(i + 1, len(sorted_group)):
                    rejected = sorted_group.iloc[j]
                    if chosen[attr] >= rejected[attr] + 1:
                        pair = {
                            'prompt': prompt,
                            'chosen': chosen['content'],
                            'rejected': rejected['content'],
                            'attribute': attr,
                            'chosen_rating':chosen[attr],
                            'rejected_rating':rejected[attr],
                        }
                        test_set.append(pair)
                        cnt_pair_wise[attr] += 1

    print("Test:",cnt_pair_wise)
    pairwise_df_train = pd.DataFrame(training_set)
    pairwise_df_test = pd.DataFrame(test_set)

    train_dataset = Dataset.from_list(training_set)
    test_dataset = Dataset.from_list(test_set)

    dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

    dataset.save_to_disk("./dataset/ultrafeedback_per_attribute_pairwise")
    # dataset.push_to_hub("pair/ultrafeedback_per_attribute_pairwise")

    return pairwise_df_train, pairwise_df_test



def create_pairwise_dataset_rpr(add_criterion = True):
    ds = load_dataset('microsoft/rpr')
    if add_criterion:
        processed_ds = []
        for item in ds['train']:
            # prompt = f'[criteria] {item["criteria_x"]}\n[context] {item["prompt"]}'
            prompt = f'{item["prompt"]} {item["criteria_x"]}'
            processed_ds.append({'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_a']}],
                                'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_b']}],
                                'attribute':'rpr-' + item["category_x"].lower().replace(' ', '-').replace('&', 'and')})
            #prompt = f'[criteria] {item["criteria_y"]}\n[context] {item["prompt"]}'
            prompt = f'{item["prompt"]} {item["criteria_y"]}'
            processed_ds.append({'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_a']}],
                                'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_b']}],
                                'attribute':'rpr-' + item["category_y"].lower().replace(' ', '-').replace('&', 'and')})
            
        processed_ds = Dataset.from_list(processed_ds)
        pairwise_df_train = pd.DataFrame(processed_ds)
        attribute_counts = pairwise_df_train['attribute'].value_counts()
        print("Train:",attribute_counts)

        test_processed_ds = []
        for item in ds['test']:
            # prompt = f'[criteria] {item["criteria_x"]}\n[context] {item["prompt"]}'
            prompt = f'{item["prompt"]} {item["criteria_x"]}'
            test_processed_ds.append({'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_a']}],
                                'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_b']}],
                                'attribute':'rpr-' + item["category_x"].lower().replace(' ', '-').replace('&', 'and')})
            # prompt = f'[criteria] {item["criteria_y"]}\n[context] {item["prompt"]}'
            prompt = f'{item["prompt"]} {item["criteria_y"]}'
            test_processed_ds.append({'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_a']}],
                                'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['response_b']}],
                                'attribute':'rpr-' + item["category_y"].lower().replace(' ', '-').replace('&', 'and')})
            
        test_processed_ds = Dataset.from_list(test_processed_ds)
        pairwise_df_test = pd.DataFrame(test_processed_ds)
        attribute_counts = pairwise_df_test['attribute'].value_counts()
        print("Test:",attribute_counts)

        processed_ds = DatasetDict({'train': processed_ds, 'test': test_processed_ds})
        processed_ds.save_to_disk("./dataset/rpr_per_category_pairwise_add_criterion_template2")
        return pairwise_df_train, pairwise_df_test
    else:
        processed_ds = []
        for item in ds['train']:
            processed_ds.append({'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_a']}],
                                'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_b']}],
                                'criteria': item['criteria_x'],
                                'attribute':'rpr-' + item["category_x"].lower().replace(' ', '-').replace('&', 'and')})
            processed_ds.append({'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_a']}],
                                'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_b']}],
                                'criteria': item['criteria_y'],
                                'attribute':'rpr-' + item["category_y"].lower().replace(' ', '-').replace('&', 'and')})
            
        processed_ds = Dataset.from_list(processed_ds)
        pairwise_df_train = pd.DataFrame(processed_ds)
        attribute_counts = pairwise_df_train['attribute'].value_counts()
        print("Train:",attribute_counts)

        test_processed_ds = []
        for item in ds['test']:
            test_processed_ds.append({'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_a']}],
                                'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_b']}],
                                'criteria': item['criteria_x'],
                                'attribute':'rpr-' + item["category_x"].lower().replace(' ', '-').replace('&', 'and')})
            test_processed_ds.append({'rejected': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_a']}],
                                'chosen': [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['response_b']}],
                                'criteria': item['criteria_y'],
                                'attribute':'rpr-' + item["category_y"].lower().replace(' ', '-').replace('&', 'and')})
            
        test_processed_ds = Dataset.from_list(test_processed_ds)
        pairwise_df_test = pd.DataFrame(test_processed_ds)
        attribute_counts = pairwise_df_test['attribute'].value_counts()
        print("Test:",attribute_counts)

        processed_ds = DatasetDict({'train': processed_ds, 'test': test_processed_ds})
        processed_ds.save_to_disk("./dataset/rpr_per_category_pairwise")
        return pairwise_df_train, pairwise_df_test
    
def create_pairwise_dataset_rlhf_hh():
    ds = load_dataset("Anthropic/hh-rlhf", data_dir='helpful-base')['train'].shuffle(seed=0)
    print('Train Helpfulness:',len(ds))
    processed_ds = []
    
    for item in tqdm(ds, desc="Processing dataset"):
        # source.append(example['subset'])
        
        chosen = item['chosen']
        rejected = item['rejected']
        chosen_splits = chosen.split('\n\nHuman: ')[1:]
        rejected_splits = rejected.split('\n\nHuman: ')[1:]

        verified = True
        chosen_turns = []
        rejected_turns = []
        for chosen_split in chosen_splits:
            pair = chosen_split.split('\n\nAssistant: ')
            if len(pair) != 2:
                verified = False
                break
            chosen_turns.append({'content': pair[0], 'role': 'user'})
            chosen_turns.append({'content': pair[1], 'role': 'assistant'})
        for rejected_split in rejected_splits:
            pair = rejected_split.split('\n\nAssistant: ')
            if len(pair) != 2:
                verified = False
                break
            rejected_turns.append({'content': pair[0], 'role': 'user'})
            rejected_turns.append({'content': pair[1], 'role': 'assistant'})
        if verified:
            example = {'chosen': chosen_turns, 'rejected': rejected_turns,'attribute':'rlhf-hh-helpfulness'}
            processed_ds.append(example)
        else:
            continue
    
    ds = load_dataset("Anthropic/hh-rlhf", data_dir='harmless-base')['train'].shuffle(seed=0)
    print('Train Harmlessness:',len(ds))
    for item in tqdm(ds, desc="Processing dataset"):
        # source.append(example['subset'])
        
        chosen = item['chosen']
        rejected = item['rejected']
        chosen_splits = chosen.split('\n\nHuman: ')[1:]
        rejected_splits = rejected.split('\n\nHuman: ')[1:]

        verified = True
        chosen_turns = []
        rejected_turns = []
        for chosen_split in chosen_splits:
            pair = chosen_split.split('\n\nAssistant: ')
            if len(pair) != 2:
                verified = False
                break
            chosen_turns.append({'content': pair[0], 'role': 'user'})
            chosen_turns.append({'content': pair[1], 'role': 'assistant'})
        for rejected_split in rejected_splits:
            pair = rejected_split.split('\n\nAssistant: ')
            if len(pair) != 2:
                verified = False
                break
            rejected_turns.append({'content': pair[0], 'role': 'user'})
            rejected_turns.append({'content': pair[1], 'role': 'assistant'})
        if verified:
            example = {'chosen': chosen_turns, 'rejected': rejected_turns,'attribute':'rlhf-hh-harmlessness'}
            processed_ds.append(example)
        else:
            continue 
    
    processed_ds = Dataset.from_list(processed_ds)


    test_processed_ds = []
    
    ds = load_dataset("Anthropic/hh-rlhf", data_dir='helpful-base')['test'].shuffle(seed=0)
    print('Test Helpfulness:',len(ds))
    for item in tqdm(ds, desc="Processing dataset"):
        # source.append(example['subset'])
        
        chosen = item['chosen']
        rejected = item['rejected']
        chosen_splits = chosen.split('\n\nHuman: ')[1:]
        rejected_splits = rejected.split('\n\nHuman: ')[1:]

        verified = True
        chosen_turns = []
        rejected_turns = []
        for chosen_split in chosen_splits:
            pair = chosen_split.split('\n\nAssistant: ')
            if len(pair) != 2:
                verified = False
                break
            chosen_turns.append({'content': pair[0], 'role': 'user'})
            chosen_turns.append({'content': pair[1], 'role': 'assistant'})
        for rejected_split in rejected_splits:
            pair = rejected_split.split('\n\nAssistant: ')
            if len(pair) != 2:
                verified = False
                break
            rejected_turns.append({'content': pair[0], 'role': 'user'})
            rejected_turns.append({'content': pair[1], 'role': 'assistant'})
        if verified:
            example = {'chosen': chosen_turns, 'rejected': rejected_turns,'attribute':'rlhf-hh-helpfulness'}
            test_processed_ds.append(example)
        else:
            continue
    
    ds = load_dataset("Anthropic/hh-rlhf", data_dir='harmless-base')['test'].shuffle(seed=0)
    print('Test Harmlessness:',len(ds))
    for item in tqdm(ds, desc="Processing dataset"):
        # source.append(example['subset'])
        
        chosen = item['chosen']
        rejected = item['rejected']
        chosen_splits = chosen.split('\n\nHuman: ')[1:]
        rejected_splits = rejected.split('\n\nHuman: ')[1:]

        verified = True
        chosen_turns = []
        rejected_turns = []
        for chosen_split in chosen_splits:
            pair = chosen_split.split('\n\nAssistant: ')
            if len(pair) != 2:
                verified = False
                break
            chosen_turns.append({'content': pair[0], 'role': 'user'})
            chosen_turns.append({'content': pair[1], 'role': 'assistant'})
        for rejected_split in rejected_splits:
            pair = rejected_split.split('\n\nAssistant: ')
            if len(pair) != 2:
                verified = False
                break
            rejected_turns.append({'content': pair[0], 'role': 'user'})
            rejected_turns.append({'content': pair[1], 'role': 'assistant'})
        if verified:
            example = {'chosen': chosen_turns, 'rejected': rejected_turns,'attribute':'rlhf-hh-harmlessness'}
            test_processed_ds.append(example)
        else:
            continue 
    
    test_processed_ds = Dataset.from_list(test_processed_ds)

    processed_ds_all = DatasetDict({'train': processed_ds, 'test': test_processed_ds})
    processed_ds_all.save_to_disk("./dataset/anthropic_rlhf_hh_pairwise")


def create_pairwise_dataset_700K(sample_size=400000):
    ds = load_dataset("hendrydong/preference_700K")['train'].shuffle(seed=42)
    processed_ds = ds.select(range(sample_size))
    test_processed_ds = ds.select(range(sample_size, sample_size + 50000,1))
    
    processed_ds_all = DatasetDict({'train': processed_ds, 'test': test_processed_ds})
    processed_ds_all.save_to_disk("./dataset/400K_pairwise")
    
def create_pairwise_dataset_reward_bench():
    ds = load_dataset("allenai/reward-bench")['filtered'].shuffle(seed=42)
    
    processed_ds_all = DatasetDict({'train': ds})
    processed_ds_all.save_to_disk("./dataset/reward_bench_pairwise")
    subset = ds['subset']
    with open('SemiMultiRM_embeddings_Gemma_2B_rewardmodel_baseline_reward_bench_source.pkl','wb') as file:
        pickle.dump(subset,file)
    
def create_pairwise_dataset_reward_bench_add_criterion():
    ds = load_dataset("allenai/reward-bench")['filtered'].shuffle(seed=42)
    processed_ds = []
    for item in ds:
        criterion = REWARDBENCH_CONTEXT_MAP[item['subset']]
        prompt = f'[criteria] {criterion}\n[context] {item["prompt"]}'
        # prompt = f'{item["prompt"]} {item["criteria_x"]}'
        processed_ds.append({'chosen': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['chosen']}],
                            'rejected': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': item['rejected']}],
                            'subset':item['subset']})
    processed_ds = Dataset.from_list(processed_ds)
    processed_ds_all = DatasetDict({'train': processed_ds})
    processed_ds_all.save_to_disk("./dataset/reward_bench_pairwise_augmented_context")
    
def create_pairwise_dataset_pku_alignment_safe():
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")['train'].shuffle(seed=42)
    print(len(ds))
    training_set=[]
    for idx, item in tqdm(enumerate(ds)):
        if item['is_response_0_safe'] is True and item['is_response_1_safe'] is True:
            continue
        prompt = item['prompt']
        responses = [item['response_0'],item['response_1']]
        chosen = responses[item['safer_response_id']]
        rejected = responses[1-item['safer_response_id']]
        training_set.append({
            'prompt':prompt,
            'chosen':chosen,
            'rejected':rejected,
            'attribute':'harmlessness'
        })
    print(len(training_set))
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")['test']
    test_set=[]
    for idx, item in tqdm(enumerate(ds)):
        if item['is_response_0_safe'] is True and item['is_response_1_safe'] is True:
            continue
        prompt = item['prompt']
        responses = [item['response_0'],item['response_1']]
        chosen = responses[item['safer_response_id']]
        rejected = responses[1-item['safer_response_id']]
        test_set.append({
            'prompt':prompt,
            'chosen':chosen,
            'rejected':rejected,
            'attribute':'harmlessness'
        })
    print(len(test_set))
    train_dataset = Dataset.from_list(training_set)
    test_dataset = Dataset.from_list(test_set)

    dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})
    dataset.save_to_disk("./dataset/pku_alignment_safe_pairwise")
    

def create_pairwise_dataset_shp_alignment():
    ds = load_dataset("stanfordnlp/SHP")['train'].shuffle(seed=42)
    print(len(ds))
    training_set=[]
    for idx, item in tqdm(enumerate(ds)):
        if item['score_A'] <= 3 or item['score_B'] <= 3:
            continue
        prompt = item['history']
        responses = [item['human_ref_A'],item['human_ref_B']]
        chosen = responses[1-item['labels']]
        rejected = responses[item['labels']]
        training_set.append({
            'domain':item['domain'],
            'prompt':prompt,
            'chosen':chosen,
            'rejected':rejected,
        })
    print(len(training_set))
    ds = load_dataset("stanfordnlp/SHP")['test']
    test_set=[]
    for idx, item in tqdm(enumerate(ds)):
        if item['score_A'] <= 3 or item['score_B'] <= 3:
            continue
        prompt = item['history']
        responses = [item['human_ref_A'],item['human_ref_B']]
        chosen = responses[1-item['labels']]
        rejected = responses[item['labels']]
        test_set.append({
            'domain':item['domain'],
            'prompt':prompt,
            'chosen':chosen,
            'rejected':rejected,
        })
    print(len(test_set))
    train_dataset = Dataset.from_list(training_set)
    test_dataset = Dataset.from_list(test_set)

    dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

    dataset.save_to_disk("./dataset/stanford_shp_pairwise")
    
# create_pairwise_dataset_shp_alignment()

# pairwise_df_train, pairwise_df_test =  create_pairwise_dataset_per_attribute_helpsteer2('nvidia/Helpsteer2')
# pairwise_df_train.to_csv('./dataset/helpsteer2_pairwise_train_per_attribute_version3.csv', index=False, encoding='utf-8-sig')
# pairwise_df_test.to_csv('./dataset/helpsteer2_pairwise_test_per_attribute_version3.csv', index=False, encoding='utf-8-sig')


# pairwise_df_train, pairwise_df_test =  create_pairwise_dataset_per_attribute_ultra('openbmb/UltraFeedback')
# pairwise_df_train.to_csv('./dataset/ultrafeedback_pairwise_train_per_attribute.csv', index=False, encoding='utf-8-sig',escapechar='\\')
# pairwise_df_test.to_csv(
#     './dataset/ultrafeedback_pairwise_test_per_attribute.csv', 
#     index=False, 
#     encoding='utf-8-sig', 
#     escapechar='\\'
# )

# pairwise_df_train, pairwise_df_test = create_pairwise_dataset_rpr(add_criterion = True)
# pairwise_df_train.to_csv('./dataset/rpr_per_category_pairwise_add_criterion.csv', index=False, encoding='utf-8-sig')
# pairwise_df_test.to_csv('./dataset/rpr_per_category_pairwise_add_criterion.csv', index=False, encoding='utf-8-sig')

pairwise_df_train, pairwise_df_test = create_pairwise_dataset_rpr(add_criterion = False)
pairwise_df_train.to_csv('./dataset/rpr_per_category_pairwise.csv', index=False, encoding='utf-8-sig')
pairwise_df_test.to_csv('./dataset/rpr_per_category_pairwise.csv', index=False, encoding='utf-8-sig')

# create_pairwise_dataset_rlhf_hh()
# create_pairwise_dataset_700K()
# create_pairwise_dataset_rpr(add_criterion = True)
# create_pairwise_dataset_reward_bench()
# create_pairwise_dataset_reward_bench_add_criterion()
# create_pairwise_dataset_pku_alignment_safe()