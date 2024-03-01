from logging import log
from typing import Iterable, Union, List, Dict, Optional, Callable, Tuple, Any

import torch
from transformers import (
        T5Tokenizer, T5ForConditionalGeneration
)
from transformers.utils.logging import set_verbosity_error

from collections import defaultdict

from itertools import chain
from re import sub

import warnings

set_verbosity_error()

class LMScorer:
    """
    Base LM scorer class intended to store models and tokenizers along
    with methods to facilitate the analysis of language model output scores.
    """
    def __init__(self, model_name: str, device: Optional[str] = 'cpu') -> None:
        """
        :param model_name: name of the model, should either be a path
            to a model (.pt or .bin file) stored locally, or a
            pretrained model stored on the Huggingface Model Hub.
        :type model_name: str
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast = True)
        self.device = device
        self.vocab = defaultdict(list)
        # {self.vocab[x.strip()].append(i) for x, i in [(self.tokenizer.decode([i]), i) for i in range(self.tokenizer.vocab_size)]}
        for i in range(self.tokenizer.vocab_size):
            decoded = [(self.tokenizer.decode(i), i)]
            for x, j in decoded:
                self.vocab[x.strip()].append(j)

    def add_special_tokens(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        raise NotImplementedError
    
    def distribution(self, batch: Iterable) -> torch.Tensor:
        raise NotImplementedError
    
    def topk(self, distribution: torch.Tensor, k: int = 1) -> Tuple:
        top_k = distribution.topk(k)
    
        probs = top_k.values.squeeze(1).exp().tolist()
        if k == 1:
            tokens = self.decode(top_k.indices.squeeze(1))
        else:
            tokens = [self.decode(x) for x in top_k.indices.squeeze(1)]
    
        return tokens, probs

    def query(self, distribution: torch.Tensor, queries: List[str]) -> Tuple:
        # this will be self.vocab tho
        query_ids = [self.vocab[a] for a in queries]
        maxlen = max(map(len, query_ids))
        query_ids = [q + [self.tokenizer.pad_token_id] * (maxlen - len(q)) if len(q) < maxlen else q for q in query_ids]
        current_batch_size = distribution.shape[0]
        probs = distribution[torch.arange(current_batch_size)[:, None], query_ids].max(1).values.exp().tolist()
        
        inv_ranks = distribution.argsort().argsort() + 1
        ranks = distribution.shape[1] - inv_ranks + 1
        token_ranks = ranks[torch.arange(current_batch_size)[:, None], query_ids].min(1).values.tolist()
    
        return probs, token_ranks

    def logprobs(self, batch: Iterable, rank: bool = False) -> Union[float, List[float]]:
        warnings.warn(
            "logprobs is deprecated, use compute_stats instead",
            DeprecationWarning
        )
        raise NotImplementedError

    def compute_stats(self, batch: Iterable, rank: bool = False) -> Union[Union[float, int], List[Union[float, int]]]:
        raise NotImplementedError

    def prepare_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        raise NotImplementedError

    def prime_text(self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]]) -> Tuple:
        raise NotImplementedError

    def token_score(self, batch: Union[str, List[str]], surprisal: bool = False, prob: bool = False, base_two: bool = False, rank: bool = False) -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        '''
        For every input sentence, returns a list of tuples in the following format:
            `(token, score)`,

        where score represents the log-probability (by default) of the token given context. Can also return ranks along with scores.

        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)

        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        '''
        raise NotImplementedError
    
    def score(self, batch: Union[str, List[str]], pool: Callable = torch.mean, *args) -> Union[float, List[float]]:
        '''
        DEPRECATED as of v 0.1.18. Check out ``sequence_score`` or ``token_score`` instead!

        Pooled estimates of sentence log probabilities, computed by the
        language model. Pooling is usually done using a function that
        is passed to the method.

        :param batch: a list of sentences that will be passed to the
            language model to score.
        :type batch: Union[str, List[str]]
        :param pool: Pooling function, is selected to be
            `torch.mean()` by default.
        :type pool: Callable

        :return: Float or list of floats specifying the log
            probabilities of the input sentence(s).
        :rtype: Union[float, List[float]]
        '''
        warnings.warn(
            "score is deprecated, use sequence_score or token_score instead",
            DeprecationWarning
        )
        result = self.logprobs(self.prepare_text(batch))
        logprob, _ = list(zip(*result))
        pooled = list(map(lambda x: pool(x, *args).tolist(), logprob))
        
        return pooled
    
    def adapt_score(self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]], pool: Callable = torch.mean, *args) -> None:
        """
        DEPRECATED as of v 0.1.18. Check out ``partial_score`` instead!
        """
        warnings.warn(
            "adapt_score is deprecated, use partial_score or token_score instead",
            DeprecationWarning
        )

    def partial_score(self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]], reduction: Callable = lambda x: x.mean(0).item(), **kwargs) -> List[float]:
        '''
        Pooled estimates of sequence log probabilities (or some modification of it), given a preamble. Pooling is usually done using a function that is passed to the method.

        :param preamble: a batch of preambles or primes passed to the
            language model. This is what the sequence is conditioned on, and the model ignores the word probabilities of this part of the input in estimating the overall score.
        :type preamble: ``Union[str, List[str]]``
        :param stimuli: a batch of sequences (same length as preamble)
            that form the main input consisting of the sequence whose
            score you want to calculate.
        :type stimuli: ``Union[str, List[str]]``
        :param reduction: Reduction function, is selected to be
            ``lambda x: x.mean(0).item()`` by default, which stands for the avg. log-probability per token for each sequence in the batch.
        :type reduction: Callable
        :param kwargs: parameters for the ``compute_stats`` call --

            * `prob` (`bool`): Whether the returned value should be a probability (note that the default reduction method will have to be changed to `lambda x: x.prod(0).item()` to get a meaningful return value)

            * `base_two` (`bool`): whether the returned value should be in base 2 (only works when `prob = False`)

            * `surprisal` (`bool`): whether the returned value should be a surprisal (does not work when `prob = True`)


        :return: List of floats specifying the desired score for the stimuli part of the input, e.g., P(stimuli | preamble).
        :rtype: ``List[float]``
        '''
        result = self.compute_stats(self.prime_text(preamble, stimuli), **kwargs, return_tensors = True)
        logprob = result
        reduced = list(map(reduction, logprob))
        
        return reduced

    def encode(self, text: Union[str, List[str]], manual_special: bool = True, return_tensors: Optional[str] = 'pt') -> Dict:
        """
        Encode a batch of sentences using the model's tokenizer.
        Equivalent of calling `model.tokenizer(input)`

        :param ``Union[str, List[str]]`` text: Input batch/sentence to
            be encoded.
        :param manual_special: Specification of whether special tokens
            will be manually encoded.
        :type manual_special: bool
        :param return_tensors: returned tensor format. Default `'pt'`
        :type manual_special: str

        :return: Encoded batch 
        :rtype: ``Dict``
        """
        sentences = [text] if isinstance(text, str) else text

        if manual_special:
            # manually add special tokens
            sentences = self.add_special_tokens(sentences)
            if return_tensors:
                tokens = self.tokenizer.batch_encode_plus(sentences, add_special_tokens = False, padding = 'longest', return_attention_mask = True, return_tensors = return_tensors)
        else:
            # mostly for masked LMs
            tokens = self.tokenizer.batch_encode_plus(sentences, padding = 'longest', return_attention_mask = True)

        return tokens
    
    def decode(self, idx: List[int]):
        """
        Decode input ids using the model's tokenizer.

        :param ``List[int]`` idx: List of ids.

        :return: Decoded strings
        :rtype: List[str]
        """
        return [self.tokenizer.decode([x]).strip() for x in self.tokenizer.convert_tokens_to_ids(self.tokenizer.convert_ids_to_tokens(idx))]
        

class Seq2SeqScorer(LMScorer):
    """
    Class for Seq2seq model like condtional T5

    :param model_name: name of the model, should either be a path
        to a model (.pt or .bin file) stored locally, or a
        pretrained model stored on the Huggingface Model Hub.
    :type model_name: str
    :param device: device type that the model should be loaded on,
        options: `cpu or cuda:{0, 1, ...}`
    :type device: str, optional
    """
    def __init__(self, model_name: str, device: Optional[str] = 'cpu') -> None:
        """
        :param model_name: name of the model, should either be a path
            to a model (.pt or .bin file) stored locally, or a
            pretrained model stored on the Huggingface Model Hub.

        :type model_name: str
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        """
        super(Seq2SeqScorer, self).__init__(model_name, device)
        
        self.model = T5ForConditionalGeneration.from_pretrained(
                model_name, return_dict = True
        )
        
        # define CLS and SEP tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
            self.tokenizer.pad_token = "<|pad|>"

        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|bos|>"]})
            self.tokenizer.bos_token = "<|bos|>"

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()
    
    def add_special_tokens(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Reformats input text to add special model-dependent tokens.

        :param text: single string or batch of strings to be
            modified.
        :type text: Union[str, List[str]]
        
        :return: Modified input, containing special tokens as per 
            tokenizer specification
        :rtype: Union[float, List[float]]:
        """
        sentences = [text] if isinstance(text, str) else text
        sentences = [self.tokenizer.bos_token + sentence for sentence in sentences]

        return sentences

    def encode(self, text: Union[str, List[str]]) -> dict:
        text = [text] if isinstance(text, str) else text
        return self.tokenizer(text, return_tensors='pt', padding = True)
    
    def prepare_text(self, text: Union[str, List[str]]) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on. 

        :param text: batch of sentences to be prepared for scoring.
        
        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        encoded = self.encode(text)
        offsets = [0] * len(encoded['input_ids'])
        return encoded, offsets
    
    def prime_text(self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]]) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on. 

        :param ``Union[str, List[str]]`` preamble: Batch of prefixes/prime/preambles on which the LM is conditioned.
        :param ``Union[str, List[str]]`` stimuli: Batch of continuations that are scored based on the conditioned text (provided in the ``preamble``). The positions of the elements match their counterparts in the ``preamble``.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        preamble_text = [preamble] if isinstance(preamble, str) else preamble
        preamble_encoded = self.tokenizer(preamble_text)['input_ids']
        preamble_lens = []
        for preamble_tokens in preamble_encoded:
            preamble_lens.append(len([token for token in preamble_tokens if token != self.tokenizer.pad_token_id and token != self.tokenizer.sep_token_id]) - 1)
        
        sentences = [preamble + " " + stimuli] if isinstance(preamble, str) else [p + " " + s for p , s in list(zip(preamble, stimuli))]
            
        return self.encode(sentences), preamble_lens
    
    def distribution(self, batch: Iterable) -> torch.Tensor:
        """
        Returns a distribution over the vocabulary of the model.

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.

        :return: Tensor consisting of log probabilies over vocab items.
        """
        batch, offsets = batch
        ids = batch["input_ids"]
        ids = ids.to(self.device)
        attention_masks = batch["attention_mask"]
        attention_masks = attention_masks.to(self.device)
        nopad_mask = ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_masks)
            logits = outputs.logits
            if self.device == 'cuda:0' or self.device == "cuda:1":
                logits.detach()

        outputs = []
        for sent_index in range(len(ids)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(batch.tokens(sent_index))
                if sent_nopad_mask[i] and i > offsets[sent_index] + 1
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            # ignore first token (<|eos|>)
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")

            outputs.append(sent_logits[-1])
        return torch.stack(outputs, 0)

    def next_word_distribution(self, queries: List, surprisal: bool = False):
        '''
        Returns the log probability distribution of the next word.
        '''
        encoded = self.encode(queries)
        encoded = encoded.to(self.device)
        query_ids = [[j for j, i in enumerate(instance) if i != self.tokenizer.pad_token_id][-1] for instance in encoded['input_ids'].tolist()]

        logits = self.model(**encoded).logits.detach()
        logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits[torch.arange(len(query_ids)), query_ids]
        logprobs = logits - logits.logsumexp(1).unsqueeze(1)

        if surprisal:
            logprobs = -1.0 * logprobs
        
        return logprobs

    def compute_stats(self, batch: Iterable, source: Iterable, rank: bool = False, prob: bool = False, base_two: bool = False, return_tensors: bool = False) -> Union[Tuple[List[float], List[float]], List[float]]:
        '''
        Primary computational method that processes a batch of prepared sentences and returns per-token scores for each sentence. By default, returns log-probabilities.

        :param ``Iterable`` batch: batched input as processed by ``prepare_text`` or ``prime_text``.
        :param ``bool`` rank: whether the model should also return ranks per word (based on the conditional log-probability of the word in context).
        :param ``bool`` prob: whether the model should return probabilities instead of log-probabilities. Can only be `True` when `base_two` is `False`.
        :param ``bool`` base_two: whether the base of the log should be 2 (usually preferred when reporting results in bits). Can only be `True` when `prob` is `False`.
        :param ``bool`` return_tensors: whether the model should return scores as a list of tensors instead of a list of lists. This is important in some other convenient methods used in the package.

        :return: Either a tuple of lists, each containing probabilities and ranks per token in each sentence passed in the input.
        :rtype: ``Union[Tuple[List[float], List[int]], List[float]]``
        '''
        assert not (base_two and prob), "cannot both use base (which is for a log), and a probability measure at the same time!"

        source_encoded, source_offsets = source
        target_encoded, target_offsets = batch
        source_ids = source_encoded['input_ids'].to(self.device)
        target_ids = target_encoded['input_ids'].to(self.device)
        
        source_ids_list = [[i for i in instance if i != self.tokenizer.pad_token_id] for instance in source_encoded['input_ids'].tolist()]
        target_ids_list = [[i for i in instance if i != self.tokenizer.pad_token_id] for instance in target_encoded['input_ids'].tolist()]

        ## Ignore the probabilities of the first token.
        source_effective_ids = [id[1:] for id in source_ids_list]
        target_effective_ids = [id[1:] for id in target_ids_list]

        with torch.no_grad():
            logits = self.model(input_ids=source_ids, labels=target_ids).logits.detach()

        logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits.split([1]*len(target_offsets))

        ## Set up storage variables
        scores = []
        if rank:
            ranks = []

        for logit, idx, offset in zip(logits, target_effective_ids, target_offsets):
            length = len(idx)
            logit = logit.squeeze(0)[:, :-1][torch.arange(offset, length),]

            logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)
            query_ids = idx[offset:]
            if base_two:
                '''
                Log_2(X) = log_e(X)/log_e(2) (broadcasted)
                '''
                score = (logprob_distribution[torch.arange(length - offset), query_ids] / torch.tensor(2).log()).tolist()
            else:
                if prob:
                    score = logprob_distribution[torch.arange(length - offset), query_ids].exp().tolist()
                else:
                    score = logprob_distribution[torch.arange(length - offset), query_ids].tolist()

            if rank:
                # shape = logprob_distribution.shape
                '''
                Double argsort trick:
                first argsort returns idxes of values that would return a sorted tensor,
                second argsort returns ranks (0 indexed)

                Proof: https://www.berkayantmen.com/rank.html

                TODO: Try to implement ranking in linear time but across arbitrary dimensions:
                https://stackoverflow.com/a/5284703
                '''
                word_ranks = (-1.0 * logprob_distribution).argsort().argsort() + 1
                # inv_ranks = logprob_distribution.argsort().argsort() + 1
                # word_ranks = shape[1] - inv_ranks + 1
                word_ranks = word_ranks[torch.arange(length - offset), query_ids].tolist()
                ranks.append(word_ranks)

            scores.append(score)

        if return_tensors:
            scores = [torch.tensor(l) for l in scores]

        if rank:
            return scores, ranks
        else:
            return scores

    def sequence_score(self, batch, reduction = lambda x: x.mean(0).item(), base_two = False,
                       source_format = 'blank', source = None):
        '''
        TODO: reduction should be a string, if it's a function, specify what kind of function. --> how to ensure it is always that type?
        '''
        if source is not None:
#             print(len(source), len(batch))
#             assert len(source) == len(batch)
            source_format = "custom"

        tokenized = self.prepare_text(batch)
        if source_format == 'blank':
            source = [""] * len(batch)
        elif source_format == 'copy':
            source = batch
        source = self.prepare_text(source)

        scores = self.compute_stats(tokenized, source, rank = False, base_two = base_two, return_tensors = True)
        reduced = list(map(reduction, scores))
        return reduced

    def token_score(self, batch: Union[str, List[str]], surprisal: bool = False, prob: bool = False, base_two: bool = False, rank: bool = False, source_format: str = 'blank') -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        '''
        For every input sentence, returns a list of tuples in the following format:
            `(token, score)`,

        where score represents the log-probability (by default) of the token given context. Can also return ranks along with scores.

        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)

        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        '''

        assert not (surprisal and prob), "cannot both evaluate probability and surprisal at the same time!"
        assert not (base_two and prob), "cannot both use base (which is for a log), and a probability measure at the same time!"

        tokenized = self.prepare_text(batch)
        if source_format == 'blank':
            source = [""] * len(batch)
        elif source_format == 'copy':
            source = batch
        source = self.prepare_text(source)

        if rank:
            scores, ranks = self.compute_stats(tokenized, source, rank = rank, prob = prob, base_two = base_two, return_tensors=True)
        else:
            scores = self.compute_stats(tokenized, source, prob = prob, base_two = base_two, return_tensors=True)

        if surprisal:
            scores = [-1.0 * s for s in scores]

        scores = [s.tolist() for s in scores]

        indices = [[i for i in indexed if i != self.tokenizer.pad_token_id] for indexed in tokenized[0]['input_ids'].tolist()]
        tokens = [self.decode(idx) for idx in indices]

        if rank:
            assert len(tokens) == len(scores) == len(ranks)
        else:
            assert len(tokens) == len(scores)

        res = []
        if rank:
            for t, s, r in zip(tokens, scores, ranks):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0]*diff + s
                    ra = [0]*diff + r
                    res.append(list(zip(t, sc, ra)))
                else:
                    res.append(list(zip(t, sc, ra)))
            # return [list(zip(t, s, r)) for t, s, r in zip(tokens, scores, ranks)]
        else:
            for t, s in zip(tokens, scores):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0]*diff + s
                    res.append(list(zip(t, sc)))
                else:
                    res.append(list(zip(t, sc)))

        return res

    def logprobs(self, batch: Iterable, rank = False, source_format: str = 'blank') -> Union[float, List[float]]:
        """
        Returns log probabilities

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.
        :param rank: Specifies whether to also return ranks of words.
        :type rank: bool

        :return: List of LM score metrics (probability and rank)
            and tokens.
        :rtype: Union[List[Tuple[torch.Tensor, str]], List[Tuple[torch.Tensor, str, int]]]
        """
        warnings.warn(
            "logprobs is deprecated, use compute_stats instead",
            DeprecationWarning
        )
        batch, offsets = batch
        ids = batch["input_ids"]
        ids = ids.to(self.device)
        attention_masks = batch["attention_mask"]
        attention_masks = attention_masks.to(self.device)
        nopad_mask = ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_masks)
            logits = outputs.logits
            if self.device == 'cuda:0' or self.device == "cuda:1":
                logits.detach()
        
        outputs = []
        for sent_index in range(len(ids)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(batch.tokens(sent_index))
                if sent_nopad_mask[i] and i > offsets[sent_index]
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            # ignore first token (<|eos|>)
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")
            # ids_scores.shape = [seq_len + 1]
            # select only the ids present in the sentence out of all vocab items (as a 2d array)
            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            # log_prob.shape = [seq_len + 1]
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)
            
            sent_log_probs = sent_log_probs.type(torch.DoubleTensor)
            sent_log_probs = sent_log_probs[offsets[sent_index]:]
            lengths = len(sent_log_probs)
            if rank:
                shape = sent_logits.shape
                inv_ranks = (sent_logits).argsort().argsort() + 1
                ranks = shape[1] - inv_ranks + 1
                word_ranks = ranks[list(range(shape[0]))[offsets[sent_index]:], sent_ids[offsets[sent_index]: ].tolist()].split(lengths)
                word_ranks = [x[0] for x in word_ranks]
                outputs.append((sent_log_probs, sent_tokens, word_ranks))
            else:
                outputs.append((sent_log_probs, sent_tokens))
            # output = (sent_log_probs.sum(), sent_ids, sent_tokens)
            # outputs.append(output)
        return outputs
