from distutils.command.config import config
import imp
from datasets import Dataset, DatasetDict
import helper
import os
import pandas as pd


class ALTDataset:

    def __init__(self, subset_languages, raw_file="data/en_yy.csv", default_split=True, random_state=42):

        assert os.path.isfile(raw_file), "Check the raw_file path or run the create_alt_dataset_script"
        config = helper.read_config()

        self.raw_file = raw_file
        self.rs = random_state
        self.default_split = default_split
        self.subset_languages = subset_languages
        self.full_dataframe = None
        self.train_df, self.eval_df, self.test_df = self.initialize_dataframes()

    def initialize_dataframes(self):
        yy_subset = self.subset_languages

        raw_df = pd.read_csv(self.raw_file, sep="|")
        clean_df = raw_df.dropna()
        self.df = clean_df.copy()
        english_df = clean_df[clean_df.lang == 'en'].drop_duplicates(["SID"])
        english_df = english_df.rename(columns={"Sent_en": "Sent_yy", "lang_en": "lang_yy"})
        dfs = [english_df]

        for lang in yy_subset:
            lang_df = clean_df[clean_df.lang == lang].copy().drop_duplicates(["SID"])
            merged_df = pd.merge(english_df, lang_df, on='SID', suffixes=("_en", f"_{lang}"))
            merged_df = merged_df.rename(columns={f"Sent_{lang}": "Sent_yy", f"lang_{lang}": "lang_yy"})
            dfs.append(merged_df)

        all_df = pd.concat(dfs, axis=0, ignore_index=True)
        all_df["URL_id"] = all_df.SID.str.split(".").str[1]

        if self.default_split:
            train_ids, test_ids, eval_ids = self.split_data()
            train_mask = all_df.URL_id.isin(train_ids)
            test_mask = all_df.URL_id.isin(test_ids)
            eval_mask = all_df.URL_id.isin(eval_ids)
            train_df = all_df[train_mask].dropna().drop(columns=['lang_en', 'URL_id'])
            test_df = all_df[test_mask].dropna().drop(columns=['lang_en', 'URL_id'])
            eval_df = all_df[eval_mask].dropna().drop(columns=['lang_en', 'URL_id'])
        else:
            eval_df = all_df.groupby("lang_yy").sample(frac=0.2, random_state=self.rs)
            test_df = eval_df.groupby("lang_yy").sample(frac=0.5, random_state=self.rs).dropna()
            train_df = all_df.drop(eval_df.index).dropna().drop(columns=['lang_en', 'URL_id'])
            eval_df = eval_df.drop(test_df.index).dropna()

        return train_df, eval_df, test_df

    def split_data(self):
        train_split = self.config['Data_Tokens']['train']
        test_split = self.config['Data_Tokens']['test']
        eval_split = self.config['Data_Tokens']['eval']

        train_ids = pd.read_csv(train_split, sep='\t', names=['id', 'x'])['id'].str.replace('URL.', '')
        test_ids = pd.read_csv(test_split, sep='\t', names=['id', 'x'])['id'].str.replace('URL.', '')
        eval_ids = pd.read_csv(eval_split, sep='\t', names=['id', 'x'])['id'].str.replace('URL.', '')

        return train_ids, test_ids, eval_ids


    def target_language_set(item):
        item["Sent_en"] = f'2{item["lang_yy"]} {item["Sent_en"]}'
        return item


    def fetch_dataset(self):
        train_ds = Dataset.from_pandas(ds.train_df).remove_columns("__index_level_0__")
        eval_ds = Dataset.from_pandas(ds.eval_df).remove_columns("__index_level_0__")
        test_ds = Dataset.from_pandas(ds.test_df).remove_columns("__index_level_0__")

        return DatasetDict({"train": train_ds, "test": test_ds, "eval": eval_ds}).map(target_language_set)


if __name__ == '__main__':
    languages = ['fil', 'vi', 'id', 'ms', 'ja', 'khm', 'th', 'hi', 'my', 'zh']
    language_file = r'/data/en_yy.csv'
    ds = ALTDataset(languages, language_file)
    alt_ds = ds.fetch_dataset()
