import os
import argparse
import tensorflow as tf
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', required=True)
parser.add_argument('--dataset_name', required=True)
args = parser.parse_args()

user_sequences = defaultdict(list)

files = [
    os.path.join(args.dataset_path, split, f)
    for split in ["training", "evaluation", "testing"]
    for f in os.listdir(os.path.join(args.dataset_path, split))
    if f.endswith('.tfrecord.gz')
]

print(f"Found {len(files)} files")

for filepath in files:
    dataset = tf.data.TFRecordDataset(filepath, compression_type='GZIP')
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        user_id = features['user_id'].int64_list.value[0]
        sequence = list(features['sequence_data'].int64_list.value)
        user_sequences[user_id].extend(sequence)

# Write output in SASRec format: one line per user
# "user_id item1 item2 item3 ..."
# output_file = f'python/data/reviews_{args.dataset_name}.txt'
# with open(output_file, 'w') as f:
#     for user_id, items in sorted(user_sequences.items()):
#         for item_id in items:
#             f.write(f"{user_id} {item_id + 1}\n")  # +1 for 1-indexing

# print(f"Written {len(user_sequences)} users to {output_file}")