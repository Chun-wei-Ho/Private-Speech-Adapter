import argparse
import pandas as pd
import os

import tqdm
from tqdm.contrib.concurrent import process_map

import subprocess
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--source-split', type=str, default='split')
    parser.add_argument('--target-split', type=str, default='filtered')
    parser.add_argument('--clips', type=str, default='data/clips')
    parser.add_argument('--word-count-max', type=int, default=5000)
    parser.add_argument('--word-count-min', type=int, default=1000)
    args = parser.parse_args()

    clips = os.path.join(args.prefix, args.lang, args.clips)
    source_split = os.path.join(args.prefix, args.lang, args.source_split)
    target_split = os.path.join(args.prefix, args.lang, args.target_split)

    train_df = pd.read_csv(os.path.join(source_split, f'{args.lang}_train.csv'))
    word_counts = dict()
    for word in tqdm.tqdm(train_df['WORD'].tolist(), desc='Count train words'):
        word_counts[word] = word_counts.get(word, 0) + 1
    selected_words = [k for k, v in word_counts.items() if v >= args.word_count_min and v <= args.word_count_max]
    # sorted_words = [k for k, v in sorted(word_counts.items(), key=lambda item: -item[1])]
    # selected_words = sorted_words[args.word_start_idx:args.word_end_idx]

    print("Filtering train data")
    filtered_train = train_df[train_df['WORD'].isin(selected_words)]
    target_folder = os.path.join(args.prefix, args.lang, args.target_split)
    os.makedirs(target_folder, exist_ok=True)
    filtered_train.to_csv(os.path.join(target_folder, f'{args.lang}_train.csv'), index=False)

    filtered_all = [filtered_train]
    for target in ['dev', 'test']:
        print(f"Filtering {target} data")
        target_df = pd.read_csv(os.path.join(source_split, f'{args.lang}_{target}.csv'))
        target_df = target_df[target_df['WORD'].isin(selected_words)]
        target_df.to_csv(os.path.join(target_folder, f'{args.lang}_{target}.csv'), index=False)
        filtered_all.append(target_df)
    filtered_all = pd.concat(filtered_all)

    with open(os.path.join(target_folder, f'word_counts.txt'), 'w') as f:
        for word in selected_words:
            count = word_counts[word]
            f.write(f'{word} {count}\n')

    cmds = []
    match = re.compile(".opus$")
    for link in tqdm.tqdm(filtered_all['LINK'].tolist(), desc='Collecting required ogg files'):
        wav_path = os.path.join(clips, link)
        new_wav_path = match.sub('.wav', wav_path)
        if not os.path.isfile(new_wav_path):
            cmds.append(['ffmpeg', '-nostdin', '-hide_banner', 
                        '-loglevel', 'error',
                        '-i', wav_path, 
                        '-acodec', 'pcm_s16le', 
                        '-ar', '8000', 
                        '-vn', new_wav_path])
    process_map(subprocess.check_call, cmds, desc='Converting ogg to wav files', chunksize=1)
