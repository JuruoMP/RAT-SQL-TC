import codecs
import copy
import logging
import os
import re
from collections import Iterable, defaultdict
from tqdm import tqdm
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union, TYPE_CHECKING

from allennlp.common import Registrable
from allennlp.common.file_utils import cached_path, FileLock
from allennlp.common.checks import ConfigurationError
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import namespace_match

logger = logging.getLogger(__name__)

DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"
NAMESPACE_PADDING_FILE = "non_padded_namespaces.txt"
_NEW_LINE_REGEX = re.compile(r"\n|\r\n")






class _TokenToIndexDefaultDict(defaultdict):
    def __init__(self, non_padded_namespaces: Set[str]) -> None:
        super().__init__(

        )


class _IndexToTokenDefaultDict(defaultdict):
    def __init__(self, non_padded_namespaces: Set[str]) -> None:
        super().__init__(
        )

class LabelVocabulary():
    """
    A Vocabulary maps strings to integers, allowing for strings to be mapped to an
    out-of-vocabulary token.

    Vocabularies are fit to a particular dataset, which we use to decide which tokens are
    in-vocabulary.

    Vocabularies also allow for several different namespaces, so you can have separate indices for
    'a' as a word, and 'a' as a character, for instance, and so we can use this object to also map
    tag and label strings to indices, for a unified :class:`~.fields.field.Field` API.  Most of the
    methods on this class allow you to pass in a namespace; by default we use the 'tokens'
    namespace, and you can omit the namespace argument everywhere and just use the default.

    This class is registered as a `Vocabulary` with four different names, which all point to
    different `@classmethod` constructors found in this class.  `from_instances` is registered as
    "from_instances", `from_files` is registered as "from_files", `from_files_and_instances` is
    registered as "extend", and `empty` is registered as "empty".  If you are using a configuration
    file to construct a vocabulary, you can use any of those strings as the "type" key in the
    configuration file to use the corresponding `@classmethod` to construct the object.
    "from_instances" is the default.  Look at the docstring for the `@classmethod` to see what keys
    are allowed in the configuration file (when there is an `instances` argument to the
    `@classmethod`, it will be passed in separately and does not need a corresponding key in the
    configuration file).

    # Parameters

    counter : `Dict[str, Dict[str, int]]`, optional (default=`None`)
        A collection of counts from which to initialize this vocabulary.  We will examine the
        counts and, together with the other parameters to this class, use them to decide which
        words are in-vocabulary.  If this is `None`, we just won't initialize the vocabulary with
        anything.

    min_count : `Dict[str, int]`, optional (default=`None`)
        When initializing the vocab from a counter, you can specify a minimum count, and every
        token with a count less than this will not be added to the dictionary.  These minimum
        counts are `namespace-specific`, so you can specify different minimums for labels versus
        words tokens, for example.  If a namespace does not have a key in the given dictionary, we
        will add all seen tokens to that namespace.

    max_vocab_size : `Union[int, Dict[str, int]]`, optional (default=`None`)
        If you want to cap the number of tokens in your vocabulary, you can do so with this
        parameter.  If you specify a single integer, every namespace will have its vocabulary fixed
        to be no larger than this.  If you specify a dictionary, then each namespace in the
        `counter` can have a separate maximum vocabulary size.  Any missing key will have a value
        of `None`, which means no cap on the vocabulary size.

    non_padded_namespaces : `Iterable[str]`, optional
        By default, we assume you are mapping word / character tokens to integers, and so you want
        to reserve word indices for padding and out-of-vocabulary tokens.  However, if you are
        mapping NER or SRL tags, or class labels, to integers, you probably do not want to reserve
        indices for padding and out-of-vocabulary tokens.  Use this field to specify which
        namespaces should `not` have padding and OOV tokens added.

        The format of each element of this is either a string, which must match field names
        exactly,  or `*` followed by a string, which we match as a suffix against field names.

        We try to make the default here reasonable, so that you don't have to think about this.
        The default is `("*tags", "*labels")`, so as long as your namespace ends in "tags" or
        "labels" (which is true by default for all tag and label fields in this code), you don't
        have to specify anything here.

    pretrained_files : `Dict[str, str]`, optional
        If provided, this map specifies the path to optional pretrained embedding files for each
        namespace. This can be used to either restrict the vocabulary to only words which appear
        in this file, or to ensure that any words in this file are included in the vocabulary
        regardless of their count, depending on the value of `only_include_pretrained_words`.
        Words which appear in the pretrained embedding file but not in the data are NOT included
        in the Vocabulary.

    min_pretrained_embeddings : `Dict[str, int]`, optional
        If provided, specifies for each namespace a minimum number of lines (typically the
        most common words) to keep from pretrained embedding files, even for words not
        appearing in the data.

    only_include_pretrained_words : `bool`, optional (default=`False`)
        This defines the strategy for using any pretrained embedding files which may have been
        specified in `pretrained_files`. If False, an inclusive strategy is used: and words
        which are in the `counter` and in the pretrained file are added to the `Vocabulary`,
        regardless of whether their count exceeds `min_count` or not. If True, we use an
        exclusive strategy: words are only included in the Vocabulary if they are in the pretrained
        embedding file (their count must still be at least `min_count`).

    tokens_to_add : `Dict[str, List[str]]`, optional (default=`None`)
        If given, this is a list of tokens to add to the vocabulary, keyed by the namespace to add
        the tokens to.  This is a way to be sure that certain items appear in your vocabulary,
        regardless of any other vocabulary computation.

    padding_token : `str`,  optional (default=`DEFAULT_PADDING_TOKEN`)
        If given, this the string used for padding.

    oov_token : `str`,  optional (default=`DEFAULT_OOV_TOKEN`)
        If given, this the string used for the out of vocabulary (OOVs) tokens.

    """

    default_implementation = "from_instances"

    def __init__(
        self,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
    ) -> None:

        self._non_padded_namespaces = set(non_padded_namespaces)

        self._token_to_index = defaultdict(int)
        self._index_to_token = defaultdict(str)    

        self._retained_counter: Optional[Dict[str, Dict[str, int]]] = None




    def add_token_to_namespace(self, token: str, namespace: str = "tokens") -> int:
        """
        Adds `token` to the index, if it is not already present.  Either way, we return the index of
        the token.
        """
        if not isinstance(token, str):
            raise ValueError(
                "Vocabulary tokens must be strings, or saving and loading will break."
                "  Got %s (with type %s)" % (repr(token), type(token))
            )
        if token not in self._token_to_index:
            index = len(self._token_to_index)
            self._token_to_index[token] = index
            self._index_to_token[index] = token
            return index
        else:
            return self._token_to_index[token]




    def add_tokens_to_namespace(self, tokens: List[str], namespace: str = "tokens") -> List[int]:
        """
        Adds `tokens` to the index, if they are not already present.  Either way, we return the
        indices of the tokens in the order that they were given.
        """
        return [self.add_token_to_namespace(token, namespace) for token in tokens]

    def get_index_to_token_vocabulary(self) -> Dict[int, str]:
        return self._index_to_token

    def get_token_to_index_vocabulary(self) -> Dict[str, int]:
        return self._token_to_index

    def set_by_index_to_token(self, d):
        for index, token in d.items():
            index = int(index)
            self._index_to_token[index] = token
            self._token_to_index[token] = index


    def get_token_from_index(self, index: int, namespace: str = "tokens") -> str:
        return self._index_to_token[index]

    def get_vocab_size(self, namespace: str = "tokens") -> int:
        return len(self._token_to_index)

    def get_namespaces(self) -> Set[str]:
        return set(self._index_to_token.keys())

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self) -> str:
        base_string = "Vocabulary with namespaces:\n"
        non_padded_namespaces = f"\tNon Padded Namespaces: {self._non_padded_namespaces}\n"
        namespaces = [
            f"\tNamespace: {name}, Size: {self.get_vocab_size(name)} \n"
            for name in self._index_to_token
        ]
        return " ".join([base_string, non_padded_namespaces] + namespaces)

    def __repr__(self) -> str:
        # This is essentially the same as __str__, but with no newlines
        base_string = "Vocabulary with namespaces: "
        namespaces = [
            f"{name}, Size: {self.get_vocab_size(name)} ||" for name in self._index_to_token
        ]
        non_padded_namespaces = f"Non Padded Namespaces: {self._non_padded_namespaces}"
        return " ".join([base_string] + namespaces + [non_padded_namespaces])




    def save_to_files(self, directory: str) -> None:
        """
        Persist this Vocabulary to files so it can be reloaded later.
        Each namespace corresponds to one file.

        # Parameters

        directory : `str`
            The directory where we save the serialized vocabulary.
        """
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            logger.warning("vocabulary serialization directory %s is not empty", directory)

        # We use a lock file to avoid race conditions where multiple processes
        # might be reading/writing from/to the same vocab files at once.
        with FileLock(os.path.join(directory, ".lock")):
            with codecs.open(
                os.path.join(directory, NAMESPACE_PADDING_FILE), "w", "utf-8"
            ) as namespace_file:
                for namespace_str in self._non_padded_namespaces:
                    print(namespace_str, file=namespace_file)

            for namespace, mapping in self._index_to_token.items():
                # Each namespace gets written to its own file, in index order.
                with codecs.open(
                    os.path.join(directory, namespace + ".txt"), "w", "utf-8"
                ) as token_file:
                    num_tokens = len(mapping)
                    start_index = 1 if mapping[0] == self._padding_token else 0
                    for i in range(start_index, num_tokens):
                        print(mapping[i].replace("\n", "@@NEWLINE@@"), file=token_file)
