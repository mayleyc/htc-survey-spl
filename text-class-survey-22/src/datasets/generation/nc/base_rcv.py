import os
import pickle
from pathlib import Path
from typing import List
from xml.dom.minidom import parse

import joblib
import numpy as np
import regex as re
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from src.datasets.generation.base import BaseDataset
from src.datasets.generation.nc.reuters_utils import ReuterArticle, parse_newsitem
from utils.split import stratified_split
from utils.visualization import labels_histogram, label_count_histogram


class BaseRCV(BaseDataset):
    """
    Generation utility for RCV1 and RCV2 datasets
    """
    MULTILABEL: bool = True

    def __init__(self, dataset_name: str, path: str = None):
        """
        Initialize RCV2 dataset generator. Convert from XML to a pickle for convenience's sake.

        Parameters
        ----------
        path : path that contains the REUT_LAN_NUM folders
        """
        super().__init__(dataset_name)

        if not path:
            path = self.RAW_ROOT / self.LANG
        # Folder containing the numbered sub-folders
        self.__folder_path: Path = Path(path)
        # List of articles formatted
        self.__articles: List[ReuterArticle] = list()
        # Labels and texts 1:1 correspondence between indexes
        self.__labels: List[List[str]] = list()
        self.__texts: List[str] = list()

        self.NUM_CLASSES = 0

    def _pickle_step(self) -> None:
        """
        Generate articles from XML files, organized in the subfolders as provided
        by NIST (e.g. rcv2it/italian/REUTIT1/<files>, rcv2it/italian/REUTIT2/<files>)
        Transform into a pickle for convenience.
        """
        # Prepare or check for pickle file
        pickle_path = self.__folder_path.parent / 'PICKLE'
        pickle_path.mkdir(exist_ok=True)
        # If pickled file doesn't exist, create it and move on
        if not (pickle_path / f'{self.LANG}.pkl').exists():
            subfolders: List[str] = os.listdir(self.__folder_path)
            paths: List[Path] = [self.__folder_path / sub for sub in subfolders]
            # Parse XML files
            for path in tqdm(paths, desc="Processing folders..."):
                for xml_file in path.iterdir():
                    dom = parse(str(xml_file))
                    newsarticle: ReuterArticle = parse_newsitem(dom)
                    if newsarticle and newsarticle.topics:
                        self.__articles.append(newsarticle)
            # Save as pickle
            with open(pickle_path / f'{self.LANG}.pkl', 'wb') as f:
                pickle.dump(self.__articles, f)
            # Sanity check
            with open(pickle_path / f'{self.LANG}.pkl', 'rb') as f:
                pickled = pickle.load(f)
            assert pickled == self.__articles, "Error: Pickled data is not equal to the one on memory"
        # If pickled file exists, just load that
        else:
            print("[INFO] Found pickled file, loading that instead.")
            with open(pickle_path / f'{self.LANG}.pkl', 'rb') as f:
                self.__articles = pickle.load(f)

    def _extract_topics(self, merge_G_categories: bool = True, plot: bool = False) -> None:
        """
        Generate dataset in FastText format.

        Parameters
        ----------
        merge_G_categories : Whether to merge G categories in a big, individual one
        plot : whether to plot and print some statistics on the extracted dataset
        """
        # Count articles that will be discarded
        uncategorized_num: int = 0
        for article in tqdm(self.__articles, desc="Extracting topics..."):
            # Find generic category
            matched_topics: List[str] = [topic for topic in article.topics if re.fullmatch(r'\wCAT', topic)]
            # Extract category code
            generic_topic_codes: List[str] = [code[:-len('CAT')] for code in matched_topics]
            # Find subcategories
            matched_subtopics: List[str] = []
            for code in generic_topic_codes:
                # Categories are {code}\d+, with d+ being 2 to 4 numbers
                # Search fetches all of them, then truncates and eliminates duplicates
                article_subtopics = [topic for topic in article.topics if re.fullmatch(rf'{code}\d+', topic)]
                article_subtopics = list(set([sub[:self.MAX_CODE_LEN] for sub in article_subtopics]))
                matched_subtopics.extend(article_subtopics)
                # Special case of G, which has non-numerical subtopics
                if code == 'G':
                    if merge_G_categories:
                        matched_subtopics.extend(['GCAT'])
                    else:
                        article_subtopics = [topic for topic in article.topics
                                             if re.fullmatch(rf'{code}[^(?:CAT)]\w+', topic)]
                        article_subtopics = list(set(article_subtopics))
                        matched_subtopics.extend(article_subtopics)
            # No categories were found
            if not matched_subtopics:
                uncategorized_num += 1
                continue
            # Minimal preprocessing of spaces in text, add headline
            _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
            text = f"{article.headline}{_RE_COMBINE_WHITESPACE.sub(' ', article.article).strip()}"
            # Add useful info
            self.__labels.append(matched_subtopics)
            self.__texts.append(text.strip())
            # --- PRINT STUFF ---
            # # All topics
            # print(article.topics)
            # # Generic topics
            # print(matched_topics)
            # # Found subtopics
            # print(matched_subtopics)
            # print(f"{' '.join([f'__label__{topic}' for topic in matched_subtopics])}{text}")
            # print("***")
            # -------------------

        if plot:
            topics, counts = np.unique(np.concatenate(self.__labels), return_counts=True)
            print(f"Topics overall ({self.LANG}): {len(topics)}")
            print(f"{uncategorized_num} articles had no suitable category.")

            labels_histogram(self.__labels, multilabel=True, title=f'{self.LANG.title()} topics '
                                                                   f'({len(topics)}, GCATS '
                                                                   f'{"" if merge_G_categories else "NOT"} condensed)')
            label_count_histogram(self.__labels, title=f"{self.LANG.title()} RCV2 number of topics per document")

    def prune(self, min_articles: int, plot: bool = False):
        """
        Removes categories with fewer than a certain number of articles. An article with both an allowed
        and disallowed category will be kept, but the category that is being removed will be removed from
        its labels.

        Parameters
        ----------
        min_articles : Minimum number of articles for a category to not be kept
        plot : Plot categories statistics
        """
        # Find populous categories
        topics, counts = np.unique(np.concatenate(self.__labels), return_counts=True)
        allowed_cats = [label for label, count in zip(topics, counts) if count > min_articles]

        current_texts = self.__texts
        self.__texts = []
        current_labels = self.__labels
        self.__labels = []
        for article_text, labels in tqdm(zip(current_texts, current_labels), desc="Pruning categories..."):
            # No suitable categories leads to article removal
            if not any(lab in allowed_cats for lab in labels):
                continue
            # If multiple categories are present but some are due for removal, remove those labels from the list
            elif not all(lab in allowed_cats for lab in labels):
                self.__texts.append(article_text)
                self.__labels.append([lab for lab in labels if lab in allowed_cats])
                assert all(lab in allowed_cats for lab in self.__labels[-1]), ("Unsuccesful removal "
                                                                               "of undesired labels")
            # Else just add
            else:
                self.__texts.append(article_text)
                self.__labels.append(labels)
        if plot:
            kept_topics = np.unique(np.concatenate(self.__labels))
            labels_histogram(self.__labels, multilabel=True, title=f'{self.LANG.title()} topics '
                                                                   f'({len(kept_topics)}, GCATS '
                                                                   f'NOT condensed, min {min_articles} articles)')
            label_count_histogram(self.__labels, title=f"{self.LANG.title()} RCV2 number of topics per document")
            # print(allowed_cats)

    def generate(self, merge_G_categories: bool = False, min_articles: int = 500, plot: bool = True) -> None:
        """
        Generate, extract topics, prune small categories and save to FastText compliant file (.txt)

        Parameters
        ----------
        merge_G_categories : Whether to merge "G" categories, which are usually very small
        min_articles : Minimum number of articles per category for a category to be worthwile
        plot : Whether to plot label statitistics in various steps
        """
        # Populates the "self.articles" by either parsing XML files or loading the pickled file
        self._pickle_step()
        # Extract topics and populates private attributes "texts" and "labels"
        self._extract_topics(merge_G_categories=merge_G_categories, plot=plot)
        # Remove categories with less than n articles
        self.prune(min_articles, plot=plot)
        # Count categories kept
        topics = np.unique(np.concatenate(self.__labels))
        self.NUM_CLASSES = len(topics)
        # Transform labels into ints, save encoders for re-conversion
        encoder = MultiLabelBinarizer()
        target = encoder.fit_transform(self.__labels)
        joblib.dump(encoder, self.ENCODER_FILEPATH)

        # Stratify splits
        x_train, x_test, y_train, y_test = stratified_split(self.__texts, target,
                                                            splits=(self.TRAIN_SIZE, 1 - self.TRAIN_SIZE))
        x_train, x_val, y_train, y_val = stratified_split(x_train, y_train, splits=(1 - self.VAL_SIZE, self.VAL_SIZE))

        y_train = encoder.inverse_transform(y_train)
        y_test = encoder.inverse_transform(y_test)
        y_val = encoder.inverse_transform(y_val)

        if plot:
            labels_histogram(y_train, title=f"{self.LANG.title()} RCV2 training topics")
            labels_histogram(y_test, title=f"{self.LANG.title()} RCV2 test topics")
            labels_histogram(y_val, title=f"{self.LANG.title()} RCV2 validation topics")
        self._save_to_txt(x_train=x_train, x_test=x_test, x_val=x_val,
                          y_train=y_train, y_test=y_test, y_val=y_val)
