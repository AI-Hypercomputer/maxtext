import io
import glob
import sentencepiece as spm
from tqdm.auto import tqdm

input_sentence_size = 40000000

def data_loader():

    import tensorflow as tf
    import tensorflow_io as tfio

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    SPECS = {
        "text": tf.TensorSpec(tf.TensorShape([]), tf.string, name="text"),
    }

    # input_sentence_size = 50000000

    import random

    random.seed(42)

    zh_data_paths = sorted(
        glob.glob("/home/genggui001/gdrive/gg-nlp-lm-new/gg_others_shuffle/**/*.jsonl.gz", recursive=True)
    )
    random.shuffle(zh_data_paths)

    ds = tf.data.TextLineDataset(
        zh_data_paths,
        compression_type="GZIP",
        buffer_size=8 * 1024 * 1024,
        num_parallel_reads=AUTOTUNE,
    )

    def decode(x):
        x = tfio.experimental.serialization.decode_json(x, specs=SPECS)
        return x['text']

    ds = ds.map(decode, num_parallel_calls=AUTOTUNE)
    ds = ds.take(int(input_sentence_size * 1.2))

    for item in tqdm(ds.as_numpy_iterator()):
        yield item

# Loads model from URL as iterator and stores the model to BytesIO.
model = io.BytesIO()

spm.SentencePieceTrainer.train(
    sentence_iterator=data_loader(), 
    model_prefix='spm_model/other',
    model_type="BPE",
    vocab_size=100000,
    self_test_sample_size=0,
    character_coverage=0.99995,
    input_sentence_size=input_sentence_size,
    seed_sentencepiece_size=1000000,
    shrinking_factor=0.75,
    num_threads=80,
    num_sub_iterations=2,
    max_sentence_length=4192,
    shuffle_input_sentence=True,
    max_sentencepiece_length=16,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    treat_whitespace_as_suffix=False,
    split_digits=True,
    vocabulary_output_piece_score=True,
    hard_vocab_limit=True,
    use_all_vocab=False,
    byte_fallback=True,
    required_chars="",
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=-1,
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>",
    pad_piece="<pad>",
    train_extremely_large_corpus=True,
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
)

# Serialize the model as file.
with open('./spm_model/other.model', 'wb') as f:
    f.write(model.getvalue())