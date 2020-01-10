"""
Robot Semantics
Generic utils for Vision-Language processing and knowledge modeling.
"""
import os
import re
import difflib
import sys
from collections import Counter
import operator
if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

import numpy as np
import owlready2 as owl

# ------------------------------------------------------------
# Functions for NLP, vocabulary, word tokens processing
# ------------------------------------------------------------

class Vocabulary(object):
    """Simple vocabulary wrapper.
    """
    def __init__(self, 
                 start_word='<sos>',
                 end_word='<eos>',
                 unk_word=None):
        # Store word_index
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word_counts = {}

        # Add special tokens
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        for special_token in [start_word, end_word, unk_word]:
            if special_token is not None:
                self.add_word(special_token)

    def __call__(self, 
                 word):
        if not word in self.word2idx:
            if self.unk_word is None:
                return None   # Return None if no unknown word's defined
            else:
                return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, 
                 word, 
                 freq=None):
        """Add individual word to vocabulary.
        """
        if not word in self.word2idx and word is not None:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        if freq is not None:
            self.word_counts[word] = freq
        else:
            self.word_counts[word] = 1

    def get_bias_vector(self):
        """Calculate bias vector from word frequency distribution.
        NOTE: Frequency need to be properly stored.
        From NeuralTalk.
        """
        words = sorted(self.word2idx.keys())
        bias_vector = np.array([1.0*self.word_counts[word] for word in words])
        bias_vector /= np.sum(bias_vector) # Normalize to frequencies
        bias_vector = np.log(bias_vector)
        bias_vector -= np.max(bias_vector) # Shift to nice numeric range
        return bias_vector

def build_vocab(texts, 
                frequency=None, 
                filters='!"#$%&()*+.,-/:;=?@[\]^`{|}~ ',
                lower=True,
                split=" ", 
                start_word='<sos>',
                end_word='<eos>',
                unk_word=None):
    """Build vocabulary over texts/captions from training set.
    """
    # Load annotations
    counter = Counter()
    for i, text in enumerate(texts):
        tokens = word_tokenize(text, filters, lower, split)
        #print(tokens)
        counter.update(tokens)
        if (i+1) % 5000 == 0:
            print('{} captions tokenized...'.format(i+1))
    print('Done.')

    # Filter out words lower than the defined frequency
    if frequency is not None:
        counter = {word: cnt for word, cnt in counter.items() if cnt >= frequency}
    else:
        counter = counter

    # Create a vocabulary warpper
    vocab = Vocabulary(start_word=start_word,
                       end_word=end_word,
                       unk_word=unk_word)

    words = sorted(counter.keys())
    for word in words:
        vocab.add_word(word, counter[word])
    return vocab

def get_maxlen(texts):
    """Calculate the maximum document length for a list of texts.
    """
    return max([len(x.split(" ")) for x in texts])

def word_tokenize(text,
                  filters='!"#$%&()*+.,-/:;=?@[\]^`{|}~ ',
                  lower=True, 
                  split=" "):
    """Converts a text to a sequence of words (or tokens).
    """
    if lower:
        text = text.lower()
    text = text.translate(maketrans(filters, split * len(filters)))
    seq = text.split(split)
    return [i for i in seq if i]

def text_to_sequence(text,
                     vocab,
                     filters='!"#$%&()*+.,-/:;=?@[\]^`{|}~ ',
                     lower=True, 
                     split=" "):
    """Convert a text to numerical sequence.
    """
    tokens = word_tokenize(text, filters, lower, split)
    seq = []
    for token in tokens:
        word_index = vocab(token)
        if word_index is not None:  # Filter out unknown words
            seq.extend([word_index])
    return seq

def sequence_to_text(seq, 
                     vocab, 
                     filter_specials=True, 
                     specials=['<pad>', '<sos>', '<eos>']):
    """Restore sequence back to text.
    """
    tokens = []
    for idx in seq:
        tokens.append(vocab.idx2word.get(idx))
    if filter_specials:
        tokens =  filter_tokens(tokens, specials)
    return ' '.join(tokens)

def texts_to_sequences(texts,
                       vocab,
                       filters='!"#$%&()*+.,-/:;=?@[\]^`{|}~ ',
                       lower=True, 
                       split=" "):
    """Wrapper to convert batch of texts to sequences.
    """
    seqs = []
    for text in texts:
        seqs.append(text_to_sequence(text, vocab, filters, lower, split))
    return np.array(seqs)

def filter_tokens(tokens, 
                  specials=['<pad>', '<sos>', '<eos>']):
    """Filter specified words.
    """
    filtered = []
    for token in tokens:
        if token not in specials:
            filtered.append(token)
    return filtered

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """(Same from Tensorflow) Pads sequences to the same length.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


# ------------------------------------------------------------
# Functions for streamline video ops
# ------------------------------------------------------------

class StreamlineVideoQueue(object):
    """Wrapper to hold images coming from a video feed.
    """
    def __init__(self,
                 window_size,
                 retrieval_limit=None):
        # Maximum queue size
        self.window_size = window_size

        # A resonable retrieval limit should be half of the maximum queue size
        self.retrieval_limit = int(np.floor(self.window_size / 2)) if retrieval_limit is None else retrieval_limit

        # Number of maximum frames needed for retrieval should not exceed queue window size
        assert self.retrieval_limit < self.window_size

        # Main queue object and some numbers to keep track of
        self.queue = []
        self.num_frames = 0
        self.num_queued = 0

    def update(self, 
               frame):
        """Add new frame into the queue. Oldest frame is removed if exceeding.
        """
        self.queue.append(frame)
        self.num_frames += 1

        if len(self.queue) > self.window_size:  # Count number of pops when we first fill the queue
            self.queue.pop(0)
            self.num_queued += 1

    def retrieve_clip(self, 
                      forced_retrieve=False):
        """Retrieve a single video clip unit.
        """
        # Force to retrieve a clip if necessary
        if forced_retrieve:
            return self.queue.copy()
        # Cannot retrieve until filling the queue
        if self.num_frames < self.window_size:
            return None
        # Allow retrieving after filling the queue first time, unless forced
        else:
            # Two situations for a legal retrieval
            # 1: The first time queue is filled
            # 2: Every time the number of incoming new frames gets to the maximum of frames needed(defined by retrieval_limit)
            if self.num_queued % self.retrieval_limit == 0:
                return self.queue.copy()

    def __len__(self):
        return len(self.queue)


# ------------------------------------------------------------
# Functions for ontology manipulation
# ------------------------------------------------------------

# Default type2str from owlready2
type2str_restriction = owl.class_construct._restriction_type_2_label

def init_onto(onto_path):
    """Helper: read ontology file & run default reasoner.
    """
    onto_path = 'file://' + onto_path
    onto = owl.get_ontology(onto_path).load()
    print('Loaded owl file at:', onto_path)
    owl.sync_reasoner()
    return onto

def _process_entity(entity, job_name, orig_entity, graph):
    """Helper: Append entity for the specified job.
    """
    edge = (orig_entity, job_name, entity)
    if edge not in graph:
        graph.append(edge)
    return graph

def _process_restriction(restriction, entity, graph):
    """Helper: Append restriction.
    """
    assert restriction.__module__ == 'owlready2.class_construct'
    
    # Grab object_property --type--> value
    object_property, value = restriction.property, restriction.value
    restriction_type = type2str_restriction[restriction.type]
    
    # Separate logical or for 'only'
    if restriction_type == 'only':
        for or_value in value.Classes:
            edge = (entity, '{},{}'.format(object_property.name, restriction_type), or_value)
            if edge not in graph:
                graph.append(edge)
            
    # No more processing for 'some'
    else:
        edge = (entity, '{},{}'.format(object_property.name, restriction_type), value)
        if edge not in graph:
            graph.append(edge)
        
    return graph

def _process_subclasses(entity, graph):
    """Helper: Append subclasses.
    """
    # Safely grab all subclasses
    try:
        subclses = list(entity.subclasses())
    except:
        subclses = []

    for subcls in subclses:
        if (entity, 'has_subclass', subcls) not in graph:
            graph.append((entity, 'has_subclass', subcls))
        if (subcls, 'subclass_of', entity) not in graph:
            graph.append((subcls, 'subclass_of', entity))

    return graph

def _populate_subclass_rel(graph):
    """Helper: Ensure 'subclass_of' and 'has_subclass' always appear in pairs.
    """
    for edge in graph:
        if edge[1] == 'subclass_of' and (edge[2], 'has_subclass', edge[0]) not in graph:
            graph.append((edge[2], 'has_subclass', edge[0]))
        elif edge[1] == 'has_subclass' and (edge[2], 'subclass_of', edge[0]) not in graph:
            graph.append((edge[2], 'subclass_of', edge[0]))
    return graph

def _process_instances(entity, graph):
    """Helper: Append individuals.
    """
    # Safely grab all individuals
    try:
        instances = entity.instances()
    except:
        instances = []

    for instance in instances:
        if instance.is_a[0] == entity:
            if (entity, 'has_individual', instance) not in graph:
                graph.append((entity, 'has_individual', instance))

    return graph

def generate_knowledge_graph(entity):
    """Helper function to grab entity-relation from onto and 
    return as knowledge graph.
    """
    graph = []

    # Part 1: Append subclasses
    graph = _process_subclasses(entity, graph)

    # Part 2: Collect equivalent_to
    try:
        equivalent_to_list = entity.INDIRECT_equivalent_to  # NOTE: Weird bug here, have to use INDIRECT
    except:
        equivalent_to_list = []
    for et in equivalent_to_list:
        # equivalent_to AND objects:
        if et.__module__ == 'owlready2.class_construct':
            for x in et.Classes:
                # For class restriction, retrieve relevant infos inside
                if x.__module__ == 'owlready2.class_construct':
                    graph = _process_restriction(x, entity, graph)
                    
    # Part 3: Look into is_a
    is_a_list = entity.is_a
    for x in is_a_list:
        # Entity: is_a indicates subclasses
        if x.__module__ == 'owlready2.entity':
            graph = _process_entity(x, 'subclass_of', entity, graph)
                
        # Restriction
        elif x.__module__ == 'owlready2.class_construct':
            graph = _process_restriction(x, entity, graph)
        
    # Part 4: Look into instances
    graph = _process_instances(entity, graph)
    
    # Part 5: Some additional filters
    graph = _populate_subclass_rel(graph)
    
    return graph

def _filter_graph(graph, onto):
    """Helper: filter graph from some ill-logical entries.
    """
    filtered_graph = []
    # Grab all individuals
    individuals = list(onto.individuals())

    for edge in graph:
        passed = True
        # Ill-logical individuals
        if edge[0] in individuals:
            passed = False
        if passed:
            filtered_graph.append(edge)
    return filtered_graph

def keyword_search_onto(keyword, onto):
    """Search and index key entity from onto given keyword.
    """
    classes = list(onto.classes())
    classes_str = [x.name for x in classes]
    all_res = difflib.get_close_matches(keyword, classes_str)
    # Only grab the most probable search keyword
    if len(all_res) > 0:
        res = all_res[0]
        return classes[classes_str.index(res)]
    else:
        return None

def _to_string(graph):
    """Helper: Convert everything collected inside graph list into
    string.
    """
    for i in range(len(graph)):
        edge = list(graph[i])
        for k in range(len(edge)):
            if type(edge[k]) is not str:
                edge[k] = edge[k].name
            edge[k] = edge[k].replace(',', ', ')
        graph[i] = (edge[0], edge[1], edge[2])
    return graph

def ontograf_simple(orig_entity, onto):
    """Interface func to search and retrieve infor for a given
    entity inside onto.
    """
    if orig_entity is None:
        return []
    
    # Initial graph search
    graph = generate_knowledge_graph(orig_entity)
    
    # Prep for other key entities given the initial graph
    entities = []
    for edge in graph:
        entities.append(edge[2])

    # 1st-level of filters, append more info from children and parent nodes
    for entity in entities:
        sub_graph = generate_knowledge_graph(entity)
        for edge in sub_graph:
            if edge[2] == orig_entity:
                if (entity, edge[1], orig_entity) not in graph and entity != orig_entity:
                    graph.append((entity, edge[1], orig_entity))

    # 2nd-level of filters, filter some ill-logical nodes
    graph = _filter_graph(graph, onto)

    # Convert everything inside graph into str
    graph = _to_string(graph)

    return graph


# ------------------------------------------------------------
# Helper Functions to Process Knowledge Graph List
# ------------------------------------------------------------

def sentence_to_graph(sentence, 
                      token2tag,
                      filters='!"#$%&()*+.,-/:;=?@[\]^`{|}~ ',
                      lower=True, 
                      split=" "):
    """Convert a command sentence into str graph form.
    Graph: [(#node1, #relation, #node2), ...]
    """
    graph = []
    tokens = word_tokenize(sentence, filters, lower, split)
    tokens_new = [_match_onto(x) for x in tokens]

    for i in range(len(tokens)):
        tag = token2tag.get(tokens[i])
        if tag is not None:
            if tag == 'ACT_NOREL':
                edge = (tokens_new[i-1], 'V2C_'+tokens[i].upper(), 'none')
            elif tag == 'ACT':
                edge = (tokens_new[i-1], 'V2C_'+tokens[i].upper(), tokens_new[i+1])
            elif tag == 'FROM':
                edge = (tokens_new[i+1], 'V2C_'+tokens[i].upper(), tokens_new[i-1])
            elif tag == 'TO':
                edge = (tokens_new[i-3], 'V2C_'+tokens[i].upper(), tokens_new[i+1])
            else:
                edge = None
            if edge is not None: graph.append(edge)
        
    return graph

def _match_onto(string):
    """Helper: Clean command sentence, match tokens to 
    entity naming inside ontology.
    """
    if string == 'humanhand':
        return 'HumanHand'
    new_string = '' + string[0].upper()
    i = 1
    while i < len(string):
        char = string[i]
        if char == '_':
            new_string += string[i+1].upper()
            i += 1
        else:
            new_string += string[i]
        i += 1
    return new_string

def retrieve_knowledge_graph(sentence,
                             token2tag,
                             onto,
                             filters='!"#$%&()*+.,-/:;=?@[\]^`{|}~ ',
                             lower=True, 
                             split=" "):
    """Given command sentence, retrieve the external knowledge graph 
    through searching from ontology.
    """
    tokens = word_tokenize(sentence, filters, lower, split)
    graph = []
    for token in tokens:
        tag = token2tag.get(token)
        token = token.replace('_', '')
        if tag in ['ENT_OBJ']:
            entity = keyword_search_onto(token, onto)
            graph += ontograf_simple(entity, onto)
    return graph