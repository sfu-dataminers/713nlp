# -*- coding: utf-8 -*-
from typing import List
from tqdm.auto import tqdm
from transformers import MarianTokenizer, MBart50TokenizerFast, utils
from datasets import Dataset, load_dataset
utils.logging.set_verbosity(50)

def make_tokens(asian_tokenizer: str, base_tokenizer: str, sent: List[str], create_tokens_file: bool = True, tokens_file_name: str = r'data/tokens.txt' ) -> MarianTokenizer:
    """
    Append the Asian language tokens generated using MBART model to the base tokenizer.
    
    :param asian_tokenizer (str): Tokenizer to create the tokens for the asian languages.
    :param base_tokenizer (str): Tokenizer to append the asian language tokens.
    :param sent (list): input to tokenize
    :param create_tokens_file (Bool): Boolean to make the token file or not.
    :param tokens_file_name (str): Name of the token file.
    :return: Tokenizer object.
    """
    
    base_t = MarianTokenizer.from_pretrained(base_tokenizer)
    asian_t = MBart50TokenizerFast.from_pretrained(asian_tokenizer)

    tokens_ls = list()

    for s in tqdm(sent):
        try:
            tokens = asian_t.tokenize(s)
            tokens = [a.lstrip("▁") for a in tokens]
            tokens_ls.extend(tokens)
        except Exception as e:
            print(f"Error In Tokenization: {e}")
            
    unique_tokens = list(dict.fromkeys(tokens_ls))
    if "" in unique_tokens:
        unique_tokens.remove("")

    base_t.add_tokens(unique_tokens, special_tokens=True)

    if create_tokens_file:
        with open(tokens_file_name, "w") as outfile:
            outfile.write(
                "\n".join(list(base_t.added_tokens_encoder.keys())))

    return base_t


if __name__ == '__main__':
    alt_ds = load_dataset('DeskDown/ALTDataset')
    teacher = "facebook/mbart-large-50-one-to-many-mmt"
    student = 'Helsinki-NLP/opus-mt-en-zh'
    sents = alt_ds["train"]["Sent_yy"]
    marian_tokenizer = make_tokens(asian_tokenizer=teacher, base_tokenizer=student, sent=sents)
