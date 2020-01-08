"""
Robot Semantics
Generic utils for Vision-Language processing and knowledge modeling.
"""
import os
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

def _process_entity(entity, job_name):
    """Helper: Append entity for the specified job.
    """
    return entity, job_name

def _process_restriction(restriction):
    """Helper: Append restriction.
    """
    assert restriction.__module__ == 'owlready2.class_construct'
    
    # Grab object_property --type--> value
    object_property, value = restriction.property, restriction.value
    restriction_type = type2str_restriction[restriction.type]
    
    # Things needed
    rel = '{},{}'.format(object_property.name, restriction_type)
    return value, rel

def _process_subclasses(entity, kg):
    """Helper: Append subclasses.
    """
    # Safely grab all subclasses
    try:
        subclses = list(entity.subclasses())
    except:
        subclses = []

    for subcls in subclses:
        if (entity, subcls, 'has_subclass') not in kg:
            kg.append((entity, subcls, 'has_subclass'))
        if (subcls, entity, 'subclass_of') not in kg:
            kg.append((subcls, entity, 'subclass_of'))

    return kg

def _process_instances(entity, kg):
    """Helper: Append individuals.
    """
    # Safely grab all individuals
    try:
        instances = entity.instances()
    except:
        instances = []

    for instance in instances:
        if instance.is_a[0] == entity:
            if (entity, instance, 'has_individual') not in kg:
                kg.append((entity, instance, 'has_individual'))

    return kg

def generate_knowledge_graph(entity):
    """Helper function to grab entity-relation from onto and 
    return as knowledge graph.
    """
    kg = []

    # Part 1: Append subclasses
    kg = _process_subclasses(entity, kg)

    # Part 2: Collect equivalent_to
    equivalent_to_list = entity.INDIRECT_equivalent_to  # NOTE: Weird bug here, have to use INDIRECT
    for et in equivalent_to_list:
        # equivalent_to AND objects:
        if et.__module__ == 'owlready2.class_construct':
            for x in et.__dict__['Classes']:
                # For class restriction, retrieve relevant infos inside
                if x.__module__ == 'owlready2.class_construct':
                    end_node, rel = _process_restriction(x)
                
                else:
                    end_node, rel = None, ''

                if ((entity, end_node, rel) not in kg) and \
                   (end_node is not None and len(rel) != 0):
                    kg.append((entity, end_node, rel))
                    
    # Part 3: Look into is_a
    is_a_list = entity.is_a
    for x in is_a_list:
        # Entity: is_a indicates subclasses
        if x.__module__ == 'owlready2.entity':
            end_node, rel = _process_entity(x, 'subclass_of')
                
        # Restriction
        elif x.__module__ == 'owlready2.class_construct':
            end_node, rel = _process_restriction(x)

        else:
            end_node, rel = None, ''
                    
        if ((entity, end_node, rel) not in kg) and \
           (end_node is not None and len(rel) != 0):
            kg.append((entity, end_node, rel))
            if rel == 'subclass_of' and (end_node, entity, 'has_subclass') not in kg:
                kg.append((end_node, entity, 'has_subclass'))
        
    # Part 4: Look into instances
    kg = _process_instances(entity, kg)
    
    return kg

def filter_kg(kg, onto):
    """Helper: filter KG from some ill-logical entries.
    """
    filtered_kg = []
    # Grab all individuals
    individuals = list(onto.individuals())

    for graph in kg:
        passed = True
        # Ill-logical individuals
        if graph[0] in individuals:
            passed = False
        if passed:
            filtered_kg.append(graph)
    return filtered_kg

def keyword_search_onto(keyword, onto):
    """Search and index key entity from onto given keyword.
    """
    classes = list(onto.classes())
    classes_str = [x.name for x in classes]

    # Simple search method from difflib
    res = difflib.get_close_matches(keyword, classes_str)[0]

    entity = classes[classes_str.index(res)]
    return entity

def ontograf_simple(orig_entity, onto):
    """Interface func to search and retrieve infor for a given
    entity inside onto.
    """
    # Initial KG search
    kg = generate_knowledge_graph(orig_entity)
    
    # Prep for other key entities given the initial kg
    entities = []
    for graph in kg:
        entities.append(graph[1])

    # 1st-level of filters, append more info from children and parent nodes
    for entity in entities:
        sub_kg = generate_knowledge_graph(entity)
        for graph in sub_kg:
            if graph[1] == orig_entity:
                if (entity, orig_entity, graph[2]) not in kg and entity != orig_entity:
                    kg.append((entity, orig_entity, graph[2]))

    # 2nd-level of filters, filter some ill-logical nodes
    kg = filter_kg(kg, onto)

    return kg
