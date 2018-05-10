import os
import sys
import spacy
import copy
import json
import math
import wikiwords

from utils import is_stopword, is_punc, Utils
from collections import Counter
from networkx.readwrite import json_graph


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """
        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class SpacyTokenizer():

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = kwargs.get('model', 'en')
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        nlp_kwargs = {'parser': False}
        if not {'lemma', 'pos', 'ner'} & self.annotators:
            nlp_kwargs['tagger'] = False
        if not {'ner'} & self.annotators:
            nlp_kwargs['entity'] = False
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ').replace('\t', ' ').replace('/', ' / ').strip()
        # remove consecutive spaces
        if clean_text.find('  ') >= 0:
            clean_text = ' '.join(clean_text.split())
        tokens = self.nlp.tokenizer(clean_text)
        if {'lemma', 'pos', 'ner'} & self.annotators:
            self.nlp.tagger(tokens)
        if {'ner'} & self.annotators:
            self.nlp.entity(tokens)

        data = []
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i].idx
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1].idx
            else:
                end_ws = tokens[i].idx + len(tokens[i].text)

            data.append((
                tokens[i].text,
                text[start_ws: end_ws],
                (tokens[i].idx, tokens[i].idx + len(tokens[i].text)),
                tokens[i].tag_,
                tokens[i].lemma_,
                tokens[i].ent_type_,
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={'non_ent': ''})


TOK = None


def init_tokenizer():
    global TOK
    TOK = SpacyTokenizer(annotators={'pos', 'lemma', 'ner'})


digits2w = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three',
            '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}


def replace_digits(words):
    global digits2w
    return [digits2w[w] if w in digits2w else w for w in words]


def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': replace_digits(tokens.words()),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
    }
    return output


def compute_four_lang_relation(graphs, four_lang_utils, dict1, dict2):
    d1_d2_four_lang_relation = []
    for d in dict1["words"]:
        all_word_match = []
        for q in dict2["words"]:
            try:
                d_edges = four_lang_utils.get_edges(json_graph.adjacency.adjacency_graph(graphs[d]))
                q_edges = four_lang_utils.get_edges(json_graph.adjacency.adjacency_graph(graphs[q]))
                all_word_match.append(int(four_lang_utils.asim_jac_and_dots(d_edges, q_edges) * 100))
            except KeyError:
                all_word_match.append(0)
        d1_d2_four_lang_relation.append(max(all_word_match))
    return d1_d2_four_lang_relation


def compute_four_lang_sentence_relation(graph1, graph2, four_lang_utils):
    try:
        d_edges = four_lang_utils.get_edges(json_graph.adjacency.adjacency_graph(graph1))
        q_edges = four_lang_utils.get_edges(json_graph.adjacency.adjacency_graph(graph2))
        return int(four_lang_utils.asim_jac_and_dots(d_edges, q_edges) * 100)
    except Exception:
        print(graph1, graph2)
        raise


def compute_features(d_dict, q_dict, c_dict, d_id, q_id, c_id, graphs, sentence_graphs):
    # in_q, in_c, lemma_in_q, lemma_in_c, tf
    q_words_set = set([w.lower() for w in q_dict['words']])
    in_q = [int(w.lower() in q_words_set and not is_stopword(w) and not is_punc(w)) for w in d_dict['words']]
    c_words_set = set([w.lower() for w in c_dict['words']])
    in_c = [int(w.lower() in c_words_set and not is_stopword(w) and not is_punc(w)) for w in d_dict['words']]

    q_words_set = set([w.lower() for w in q_dict['lemma']])
    lemma_in_q = [int(w.lower() in q_words_set and not is_stopword(w) and not is_punc(w)) for w in d_dict['lemma']]
    c_words_set = set([w.lower() for w in c_dict['lemma']])
    lemma_in_c = [int(w.lower() in c_words_set and not is_stopword(w) and not is_punc(w)) for w in d_dict['lemma']]

    tf = [0.1 * math.log(wikiwords.N * wikiwords.freq(w.lower()) + 10) for w in d_dict['words']]
    tf = [float('%.2f' % v) for v in tf]
    d_words = Counter(filter(lambda w: not is_stopword(w) and not is_punc(w), d_dict['words']))
    four_lang_utils = Utils()
    p_q_four_lang_relation = compute_four_lang_relation(graphs, four_lang_utils, d_dict, q_dict)
    p_c_four_lang_relation = compute_four_lang_relation(graphs, four_lang_utils, d_dict, c_dict)
    q_c_four_lang_relation = compute_four_lang_relation(graphs, four_lang_utils, q_dict, c_dict)
    p_q_four_lang_sentence_relation =\
        compute_four_lang_sentence_relation(sentence_graphs[d_id],
                                            sentence_graphs[d_id]["questions"][q_id], four_lang_utils)
    p_c_four_lang_sentence_relation =\
        compute_four_lang_sentence_relation(sentence_graphs[d_id],
                                            sentence_graphs[d_id]["questions"][q_id]["choice"][c_id],
                                            four_lang_utils)
    q_c_four_lang_sentence_relation =\
        compute_four_lang_sentence_relation(sentence_graphs[d_id]["questions"][q_id],
                                            sentence_graphs[d_id]["questions"][q_id]["choice"][c_id],
                                            four_lang_utils)
    from conceptnet import concept_net
    p_q_relation = concept_net.p_q_relation(d_dict['words'], q_dict['words'])
    p_c_relation = concept_net.p_q_relation(d_dict['words'], c_dict['words'])
    assert len(in_q) == len(in_c) and len(lemma_in_q) == len(in_q) and len(lemma_in_c) == len(in_q) and len(tf) == len(in_q)
    assert len(tf) == len(p_q_relation) and len(tf) == len(p_c_relation)
    return {
        'in_q': in_q,
        'in_c': in_c,
        'lemma_in_q': lemma_in_q,
        'lemma_in_c': lemma_in_c,
        'tf': tf,
        'p_q_relation': p_q_relation,
        'p_c_relation': p_c_relation,
        'p_q_four_lang_relation': p_q_four_lang_relation,
        'p_c_four_lang_relation': p_c_four_lang_relation,
        'q_c_four_lang_relation': q_c_four_lang_relation,
        'p_q_four_lang_sentence_relation': p_q_four_lang_sentence_relation,
        'p_c_four_lang_sentence_relation': p_c_four_lang_sentence_relation,
        'q_c_four_lang_sentence_relation': q_c_four_lang_sentence_relation
    }


def get_example(d_id, q_id, c_id, d_dict, q_dict, c_dict, label):
    return {
            'id': d_id + '_' + q_id + '_' + c_id,
            'd_words': ' '.join(d_dict['words']),
            'd_pos': d_dict['pos'],
            'd_ner': d_dict['ner'],
            'q_words': ' '.join(q_dict['words']),
            'q_pos': q_dict['pos'],
            'c_words': ' '.join(c_dict['words']),
            'label': label
        }


def preprocess_4lang_sentences(path, request_path):
    import requests
    print(path)
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    sentence_graph_results = {}
    objects = json.load(open(path, 'r', encoding='utf-8'))['data']['instance']
    for obj in objects:
        print(obj)
        try:
            sentence_graph_results[obj['@id']] = \
                requests.post(request_path, data=json.dumps({'prem': obj['text'], 'hyp': ''}),
                              timeout=60, headers=headers).json()['prem']
        except:
            import re
            split = []
            sentences = re.split('\.', obj['text'])
            for s in sentences:
                split += re.split('!', s)
            graphs = []
            for index, sent in enumerate(split[:-1]):
                graphs.append(requests.post(request_path, data=json.dumps({'prem': sent, 'hyp': split[index + 1]}),
                                            timeout=60, headers=headers).json())
            sentence_graph_results[obj['@id']] = graphs[0]['prem']
            for g in graphs:
                sentence_graph_results[obj['@id']]['nodes'] += g['prem']['nodes'] + g['hyp']['nodes']
        if 'questions' in obj and obj['questions'] is not None:
            sentence_graph_results[obj['@id']]["questions"] = {}
            if isinstance(obj['questions']['question'], list):
                for question in obj['questions']['question']:
                    sentence_graph_results[obj['@id']]["questions"][question['@id']] = \
                        requests.post(request_path, data=json.dumps({'prem': question['@text'], 'hyp': ''}),
                                      timeout=60, headers=headers).json()['prem']
                    sentence_graph_results[obj['@id']]["questions"][question['@id']]["choice"] = {}
                    for choice in question['answer']:
                        sentence_graph_results[obj['@id']]["questions"][question['@id']]["choice"][choice['@id']] = \
                            requests.post(request_path, data=json.dumps({'prem': choice['@text'], 'hyp': ''}),
                                          timeout=60, headers=headers).json()['prem']
            else:
                question = obj['questions']['question']
                sentence_graph_results[obj['@id']]["questions"][question['@id']] = \
                    requests.post(request_path, data=json.dumps({'prem': question['@text'], 'hyp': ''}),
                                  timeout=60, headers=headers).json()['prem']
                sentence_graph_results[obj['@id']]["questions"][question['@id']]["choice"] = {}
                for choice in question['answer']:
                    sentence_graph_results[obj['@id']]["questions"][question['@id']]["choice"][choice['@id']] = \
                        requests.post(request_path, data=json.dumps({'prem': choice['@text'], 'hyp': ''}),
                                      timeout=60, headers=headers).json()['prem']
    try:
        with open(path.replace('.json', '') + '-4lang-' + request_path.split('/')[-1] + '.json', 'w') as \
                four_lang_sentences:
            four_lang_sentences.write(json.dumps(sentence_graph_results))
        return sentence_graph_results
    except:
        return sentence_graph_results


def preprocess_dataset(path, path_4lang_vocab, request_type, is_test_set=False):
    writer = open(path.replace('.json', '') + '-processed.json', 'w', encoding='utf-8')
    ex_cnt = 0
    with open(path_4lang_vocab, 'r') as json_graphs:
        graphs = json.loads(json_graphs.read())
        if os.path.isfile(path.replace('.json', '') + '-4lang-' + request_type.split('/')[-1] + '.json'):
            with open(path.replace('.json', '') + '-4lang-' + request_type.split('/')[-1] + '.json', 'r') as four_lang_sentences:
                sentence_graphs = json.loads(four_lang_sentences.read())
        else:
            sentence_graphs = preprocess_4lang_sentences(path, request_type)
        for obj in json.load(open(path, 'r', encoding='utf-8'))['data']['instance']:
            if not obj['questions']:
                continue
            d_dict = tokenize(obj['text'])
            d_id = path + '_' + obj['@id']
            try:
                qs = [q for q in obj['questions']['question']]
                dummy = qs[0]['@text']
            except:
                # some passages have only one question
                qs = [obj['questions']['question']]
            for q in qs:
                q_dict = tokenize(q['@text'])
                q_id = q['@id']
                for ans in q['answer']:
                    c_dict = tokenize(ans['@text'])
                    label = int(ans['@correct'].lower() == 'true') if not is_test_set else -1
                    c_id = ans['@id']
                    example = get_example(d_id, q_id, c_id, d_dict, q_dict, c_dict, label)
                    example.update(compute_features(d_dict, q_dict, c_dict, d_id.split('_')[-1], q_id, c_id, graphs, sentence_graphs))
                    writer.write(json.dumps(example))
                    writer.write('\n')
                    ex_cnt += 1
    print('Found %d examples in %s...' % (ex_cnt, path))
    writer.close()


def _get_race_obj(d):
    for root_d, _, files in os.walk(d):
        for f in files:
            if f.endswith('txt'):
                obj = json.load(open(root_d + '/' + f, 'r', encoding='utf-8'))
                yield obj


def preprocess_race_dataset(d):
    import utils
    utils.build_vocab()

    def is_passage_ok(words):
        return len(words) >= 50 and len(words) <= 500 and sum([int(w in utils.vocab) for w in words]) >= 0.85 * len(words)

    def is_question_ok(words):
        return True

    def is_option_ok(words):
        s = ' '.join(words).lower()
        return s != 'all of the above' and s != 'none of the above'
    writer = open('../data/race-processed.json', 'w', encoding='utf-8')
    ex_cnt = 0
    for obj in _get_race_obj(d):
        d_dict = tokenize(obj['article'].replace('\n', ' ').replace('--', ' '))
        if not is_passage_ok(d_dict['words']):
            continue
        d_id = obj['id']
        assert len(obj['options']) == len(obj['answers']) and len(obj['answers']) == len(obj['questions'])
        q_cnt = 0
        for q, ans, choices in zip(obj['questions'], obj['answers'], obj['options']):
            q_id = str(q_cnt)
            q_cnt += 1
            ans = ord(ans) - ord('A')
            assert 0 <= ans < len(choices)
            q_dict = tokenize(q.replace('_', ' _ '))
            if not is_question_ok(q_dict['words']):
                continue
            for c_id, choice in enumerate(choices):
                c_dict = tokenize(choice)
                if not is_option_ok(c_dict['words']):
                    continue
                label = int(c_id == ans)
                c_id = str(c_id)
                example = get_example(d_id, q_id, c_id, d_dict, q_dict, c_dict, label)
                example.update(compute_features(d_dict, q_dict, c_dict))
                writer.write(json.dumps(example))
                writer.write('\n')
                ex_cnt += 1
    print('Found %d examples in %s...' % (ex_cnt, d))
    writer.close()


def preprocess_conceptnet(path):
    import utils
    utils.build_vocab()
    writer = open('concept.filter', 'w', encoding='utf-8')

    def _get_lan_and_w(arg):
        arg = arg.strip('/').split('/')
        return arg[1], arg[2]
    for line in open(path, 'r', encoding='utf-8'):
        fs = line.split()
        relation, arg1, arg2 = fs[1].split('/')[-1], fs[2], fs[3]
        lan1, w1 = _get_lan_and_w(arg1)
        if lan1 != 'en' or not all(w in utils.vocab for w in w1.split('_')):
            continue
        lan2, w2 = _get_lan_and_w(arg2)
        if lan2 != 'en' or not all(w in utils.vocab for w in w2.split('_')):
            continue
        obj = json.loads(fs[-1])
        if obj['weight'] < 1.0:
            continue
        writer.write('%s %s %s\n' % (relation, w1, w2))
    writer.close()


def preprocess_4lang(path, vocab):
    import requests
    import os.path
    if not os.path.isfile(path):
        with open(vocab, 'r', encoding='utf-8') as vocab:
            headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
            result_vocab = {}
            line = vocab.readline().strip()
            while line is not None and line != "":
                result_vocab[line] = requests.post("http://hlt.bme.hu/4lang/definition", data=json.dumps({'word': line}), headers=headers).json()
                line = vocab.readline().strip()
        with open(path, 'w') as result_file:
            json_result = json.dumps(result_vocab)
            result_file.write(json_result)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'conceptnet':
        preprocess_conceptnet('conceptnet-assertions-5.5.5.csv')
        exit(0)
    init_tokenizer()
    path_4lang = '../data/4lang.json'
    request_paths = ['http://hlt.bme.hu/4lang/default',
                     'http://hlt.bme.hu/4lang/expand',
                     'http://hlt.bme.hu/4lang/abstract']
    data_paths = ['../data/trial-data.json',
                  '../data/dev-data.json',
                  '../data/train-data.json',
                  '../data/test-data.json']
    preprocess_4lang(path_4lang, '../data/vocab')
    for data_path in data_paths:
        # preprocess_4lang_sentences(data_path, 'http://hlt.bme.hu/4lang/default')
        if data_path.split('/')[2].startswith('test'):
            preprocess_dataset(data_path, path_4lang, 'http://hlt.bme.hu/4lang/default', is_test_set=True)
        else:
            preprocess_dataset(data_path, path_4lang, 'http://hlt.bme.hu/4lang/default')
    # preprocess_race_dataset('../data/RACE/')
